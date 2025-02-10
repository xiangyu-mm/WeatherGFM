import torch
import os.path
import sys
sys.path.append("..") 

from functools import partial
import numpy as np

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block, trunc_normal_
#from vqgan import get_vq_model
from CNN_Head import Simple_Head

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=2,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        #print('img size: ', img_size)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_4c = PatchEmbed(img_size, patch_size, 4, embed_dim)
        self.patch_embed_13c = PatchEmbed(img_size, patch_size, 13, embed_dim)
        # self.patch_embed_target = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.align_patch = nn.Linear(embed_dim, embed_dim, bias=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.patch_size = patch_size
        self.num_imgs = 4
        self.max_resolution_H = img_size
        self.max_resolution_W = img_size
        self.norm_pix_loss = False

            # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # 2 channels decoder
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # 4 channels decoder
        self.decoder_pred_4c = nn.Linear(decoder_embed_dim, patch_size**2 * 4, bias=True)
        # era5 decoder
        self.decoder_pred_era5 = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True)
        # --------------------------------------------------------------------------

        self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(48)]
            )

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------

        self.CNN_Head = Simple_Head(in_c=2, out_c=2)
        self.CNN_Head_era5 = Simple_Head(in_c=1, out_c=1)
        
        self.L1_loss = nn.L1Loss()
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        Max_num_patches = self.num_imgs*(512//self.patch_size)*(512//self.patch_size)
        
        pos_embed = get_sinusoid_encoding_table(Max_num_patches+1, self.embed_dim).cuda()
        self.class_pos_embed = pos_embed[:, 0:1, :]
        self.img1_pos_embed = pos_embed[:, 1:(512//self.patch_size)*(512//self.patch_size)+1, :]
        self.img2_pos_embed = pos_embed[:, (512//self.patch_size)*(512//self.patch_size)+1:2*(512//self.patch_size)*(512//self.patch_size)+1, :]
        self.img3_pos_embed = pos_embed[:, 2*(512//self.patch_size)*(512//self.patch_size)+1:3*(512//self.patch_size)*(512//self.patch_size)+1, :]
        self.img4_pos_embed = pos_embed[:, 3*(512//self.patch_size)*(512//self.patch_size)+1:4*(512//self.patch_size)*(512//self.patch_size)+1, :]
        
        decoder_pos_embed = get_sinusoid_encoding_table(Max_num_patches+1, self.decoder_embed_dim).cuda()
        self.class_decoder_pos_embed = decoder_pos_embed[:, 0:1, :]
        self.img1_decoder_pos_embed = decoder_pos_embed[:, 1:(512//self.patch_size)*(512//self.patch_size)+1, :]
        self.img2_decoder_pos_embed = decoder_pos_embed[:, (512//self.patch_size)*(512//self.patch_size)+1:2*(512//self.patch_size)*(512//self.patch_size)+1, :]
        self.img3_decoder_pos_embed = decoder_pos_embed[:, 2*(512//self.patch_size)*(512//self.patch_size)+1:3*(512//self.patch_size)*(512//self.patch_size)+1, :]
        self.img4_decoder_pos_embed = decoder_pos_embed[:, 3*(512//self.patch_size)*(512//self.patch_size)+1:4*(512//self.patch_size)*(512//self.patch_size)+1, :]

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        for i in range(len(self.token_embeds)):
            ws = self.token_embeds[i].proj.weight.data
            torch.nn.init.xavier_uniform_(ws.view([ws.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking_specific(self, x, mask_ratio, presever_list):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L-len(presever_list)) * (1 - mask_ratio)) + len(presever_list)
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise[:, presever_list] = 0
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x_in, mask_ratio, input_is_list=True, train_mode=True):
        if not input_is_list:
            # split input images
            B, S, C, H, W = x_in.shape
            
            x1 = x_in[:, 0, :, :, :]
            x2 = x_in[:, 1, :, :, :]
            x3 = x_in[:, 2, :, :, :]
            x4 = x_in[:, 3, :, :, :]
        else:
            x1 = x_in[0]
            x2 = x_in[1]
            x3 = x_in[2]
            x4 = x_in[3]
        
            B, C, H, W = x1.shape
        
        if C > 10:
            lead_time = x_in[4]
            out_var = x_in[5][0]

            x1_embeds = []
            for i in range(48):
                id = i
                x1_embeds.append(self.token_embeds[id](x1[:, i : i + 1]))
            x1_embed = torch.stack(x1_embeds, dim=1)  # B, V, L, D
            x3_embeds = []
            for i in range(48):
                id = i
                x3_embeds.append(self.token_embeds[id](x3[:, i : i + 1]))
            x3_embed = torch.stack(x3_embeds, dim=1)  # B, V, L, D
            x1_embed = self.aggregate_variables(x1_embed)  # B, L, D
            x3_embed = self.aggregate_variables(x3_embed)  # B, L, D

            lead_time = lead_time.float()
            lead_time_emb = self.lead_time_embed(lead_time.unsqueeze(-1))  # B, D

            lead_time_emb = lead_time_emb.unsqueeze(1)
            x1_embed = x1_embed + lead_time_emb  # B, L, D
            x3_embed = x3_embed + lead_time_emb  # B, L, D
            x2_embed = self.token_embeds[out_var](x2)
            x4_embed = self.token_embeds[out_var](x4)

        elif C == 4:
            x1_tmp = self.patch_embed_4c(x1)
            x1_embed = self.align_patch(x1_tmp)
            x3_tmp = self.patch_embed_4c(x3)
            x3_embed = self.align_patch(x3_tmp)

            x2_embed = self.patch_embed(x2)
            x4_embed = self.patch_embed(x4)
        else:
            x1_embed = self.patch_embed(x1)
            x3_embed = self.patch_embed(x3)           
        
            x2_embed = self.patch_embed(x2)
            x4_embed = self.patch_embed(x4)
        
        
        _, x1_embed_len, _ = x1_embed.shape
        _, x2_embed_len, _ = x2_embed.shape
        _, x3_embed_len, _ = x3_embed.shape
        _, x4_embed_len, _ = x4_embed.shape
        
        self.x1_embed_len = x1_embed_len
        self.x2_embed_len = x2_embed_len
        self.x3_embed_len = x3_embed_len
        self.x4_embed_len = x4_embed_len
        
        # add pos embed w/o cls token
        x1_embed = x1_embed + self.img1_pos_embed[:, :x1_embed_len, :]
        x2_embed = x2_embed + self.img2_pos_embed[:, :x2_embed_len, :]
        x3_embed = x3_embed + self.img3_pos_embed[:, :x3_embed_len, :]
        x4_embed = x4_embed + self.img4_pos_embed[:, :x4_embed_len, :]        
        
        x = torch.cat((x1_embed, x2_embed, x3_embed, x4_embed), dim=1)
                
        # masking: length -> length * mask_ratio
        preserve_list = []
        if train_mode:
            # preserve_list1 = list(range(0, x1_embed_len))
            preserve_list1 = list(range(0, x1_embed_len+x2_embed_len))
            preserve_list2 = list(range(x1_embed_len+x2_embed_len, x1_embed_len+x2_embed_len+x3_embed_len))
            preserve_list.extend(preserve_list1)
            preserve_list.extend(preserve_list2)
        else:
            preserve_list = list(range(0, x1_embed_len+x2_embed_len+x3_embed_len))
        x, mask, ids_restore = self.random_masking_specific(x, mask_ratio, preserve_list)
        
        # append cls token
        cls_token = self.cls_token + self.class_pos_embed
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, in_c, out_c):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # class token
        x[:, :1, :] = x[:, :1, :] + self.class_decoder_pos_embed
        x[:, 1:self.x1_embed_len+1, :] = x[:, 1:self.x1_embed_len+1, :] + self.img1_decoder_pos_embed[:, :self.x1_embed_len, :]
        x[:, self.x1_embed_len+1:self.x1_embed_len+self.x2_embed_len+1, :] = x[:, self.x1_embed_len+1:self.x1_embed_len+self.x2_embed_len+1, :] + self.img2_decoder_pos_embed[:, :self.x2_embed_len, :]
        x[:, self.x1_embed_len+self.x2_embed_len+1:self.x1_embed_len+self.x2_embed_len+self.x3_embed_len+1, :] = x[:, self.x1_embed_len+self.x2_embed_len+1:self.x1_embed_len+self.x2_embed_len+self.x3_embed_len+1, :] + self.img3_decoder_pos_embed[:, :self.x3_embed_len, :]
        x[:, self.x1_embed_len+self.x2_embed_len+self.x3_embed_len+1:self.x1_embed_len+self.x2_embed_len+self.x3_embed_len+self.x3_embed_len+1, :] = x[:, self.x1_embed_len+self.x2_embed_len+self.x3_embed_len+1:self.x1_embed_len+self.x2_embed_len+self.x3_embed_len+self.x3_embed_len+1, :] + self.img4_decoder_pos_embed[:, :self.x4_embed_len, :]


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if in_c > 20:
            x = self.decoder_pred_era5(x)
            # remove cls token
            x = x[:, 1:, :]
            return x
        elif in_c != out_c:
            x_2c = self.decoder_pred(x)
            x_4c  =self.decoder_pred_4c(x)
            x_2c = x_2c[:, 1:, :]
            x_4c = x_4c[:, 1:, :]
            return [x_2c, x_4c]
        else:
            x = self.decoder_pred(x)
            # remove cls token
            x = x[:, 1:, :]
            return x

    def forward_loss_pix(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = imgs
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_pix_CNN(self, pred_target_img1, pred_target_img2, target_img1, target_img2):
        
        loss = self.L1_loss(pred_target_img1, target_img1) + self.L1_loss(pred_target_img2, target_img2)
        return loss
    
    def get_patch_num(self, imgs):
        input_img1 = imgs[0]
        target_img1 = imgs[1]
        input_img2 = imgs[2]
        target_img2 = imgs[3]
        input_img1_patch_num = input_img1.shape[2]//self.patch_size*input_img1.shape[3]//self.patch_size
        target_img1_patch_num = target_img1.shape[2]//self.patch_size*target_img1.shape[3]//self.patch_size
        input_img2_patch_num = input_img2.shape[2]//self.patch_size*input_img2.shape[3]//self.patch_size
        target_img2_patch_num = target_img2.shape[2]//self.patch_size*target_img2.shape[3]//self.patch_size

        return input_img1_patch_num, target_img1_patch_num, input_img2_patch_num, target_img2_patch_num


    def forward_CNN_Head(self, pred, imgs, input_is_list=True):
        if not input_is_list:
            # split input images
            B, S, C, H, W = imgs.shape
            
            input_img1 = imgs[:, 0, :, :, :]
            target_img1 = imgs[:, 1, :, :, :]
            input_img2 = imgs[:, 2, :, :, :]
            target_img2 = imgs[:, 3, :, :, :]
        else:
            input_img1 = imgs[0]
            target_img1 = imgs[1]
            input_img2 = imgs[2]
            target_img2 = imgs[3]
        
        input_img1_patch_num = input_img1.shape[2]//self.patch_size*input_img1.shape[3]//self.patch_size
        target_img1_patch_num = target_img1.shape[2]//self.patch_size*target_img1.shape[3]//self.patch_size
        input_img2_patch_num = input_img2.shape[2]//self.patch_size*input_img2.shape[3]//self.patch_size
        
        pred_target_img1 = pred[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num, :]
        pred_target_img1 = self.unpatchify(pred_target_img1).float()
        
        pred_target_img2 = pred[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:, :]
        pred_target_img2 = self.unpatchify(pred_target_img2).float()
        
        if input_img1.shape[1] < 10:
            pred_target_img1 = self.CNN_Head(pred_target_img1)
            pred_target_img2 = self.CNN_Head(pred_target_img2)
        else:
            pred_target_img1 = self.CNN_Head_era5(pred_target_img1)
            pred_target_img2 = self.CNN_Head_era5(pred_target_img2)
        
        return pred_target_img1, pred_target_img2
    
    def forward_loss_vqgan(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        imgs = imgs[:, 0, :, :, :]
        
        with torch.no_grad():
            target = self.vae.get_codebook_indices(imgs).flatten(1)
        loss = nn.CrossEntropyLoss(reduction='none')(input=pred.permute(0, 2, 1), target=target)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

        

    def forward(self, imgs, visual_tokens=None, mask_ratio=0.75, input_is_list=True, train_mode=True):
        loss = {}

        if not input_is_list:
            # split input images
            B, S, C, H, W = imgs.shape
        else:
            x1 = imgs[0]
            x2 = imgs[1]       
            _, in_c, _, _ = x1.shape
            _, out_c, _, _ = x2.shape

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, 
                                                         input_is_list=input_is_list, train_mode=train_mode)

        # print('mask shape: ', mask.shape)
        # print('ids_restore shape: ', ids_restore.shape)
        multi_channel=False
        if in_c == 4:
            multi_channel=True
        
        pred = self.forward_decoder(latent, ids_restore, in_c, out_c)  # [N, L, p*p*3]
        if multi_channel:
            pred_target_img1, pred_target_img2 = self.forward_CNN_Head(pred[0], imgs, input_is_list=input_is_list)
        else:
            pred_target_img1, pred_target_img2 = self.forward_CNN_Head(pred, imgs, input_is_list=input_is_list)
        
        if visual_tokens is not None:
            if not isinstance(visual_tokens, list):
                target_img1 = visual_tokens[:, 1, :, :, :]
                target_img2 = visual_tokens[:, 3, :, :, :]
                
                visual_tokens1 = self.patchify(visual_tokens[:, 0, :, :, :])
                visual_tokens2 = self.patchify(visual_tokens[:, 1, :, :, :])
                visual_tokens3 = self.patchify(visual_tokens[:, 2, :, :, :])
                visual_tokens4 = self.patchify(visual_tokens[:, 3, :, :, :])

                visual_tokens = torch.cat([visual_tokens1, visual_tokens2,
                                        visual_tokens3, visual_tokens4], dim=1)
                
                loss['pix_loss'] = self.forward_loss_pix(visual_tokens, pred, mask)
                loss['pix_loss_CNN'] = self.forward_loss_pix_CNN(pred_target_img1, pred_target_img2, target_img1, target_img2)

            else:
                target_img1 = visual_tokens[1]
                target_img2 = visual_tokens[3]

                visual_tokens1 = self.patchify(visual_tokens[0])
                visual_tokens2 = self.patchify(visual_tokens[1])
                visual_tokens3 = self.patchify(visual_tokens[2])
                visual_tokens4 = self.patchify(visual_tokens[3])

                if in_c == 2:
                    visual_tokens = torch.cat([visual_tokens1, visual_tokens2,
                                            visual_tokens3, visual_tokens4], dim=1)
                    
                    loss['pix_loss'] = self.forward_loss_pix(visual_tokens, pred, mask)
                    loss['pix_loss_CNN'] = self.forward_loss_pix_CNN(pred_target_img1, pred_target_img2, target_img1, target_img2)
                elif in_c == 4:
                    visual_tokens = torch.cat([visual_tokens2, visual_tokens2,
                                            visual_tokens4, visual_tokens4], dim=1)
                    a,b,c,d = self.get_patch_num(imgs)
                                       
                    loss['pix_loss'] = self.forward_loss_pix(visual_tokens, pred[0], mask)
                    loss['pix_loss_CNN'] = self.forward_loss_pix_CNN(pred_target_img1, pred_target_img2, target_img1, target_img2)
                else:
                    visual_tokens = torch.cat([visual_tokens2, visual_tokens2,
                                            visual_tokens4, visual_tokens4], dim=1)
                    a,b,c,d = self.get_patch_num(imgs)
                    
                    loss['pix_loss'] = self.forward_loss_pix(visual_tokens, pred, mask)
                    loss['pix_loss_CNN'] = self.forward_loss_pix_CNN(pred_target_img1, pred_target_img2, target_img1, target_img2)
                
        return loss, pred, mask, pred_target_img1, pred_target_img2


def mae_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b_input256(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=256, patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

mae_vit_large_patch16_input256 = mae_vit_large_patch16_dec512d8b_input256  # decoder: 512 dim, 8 blocks


