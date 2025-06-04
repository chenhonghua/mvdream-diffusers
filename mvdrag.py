from typing import Dict, Union, Tuple, Optional
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
from mvdream_diffusers.mvdream.utils import get_camera
from mvdream_diffusers.mvdream.pipeline_mvdream import MVDreamPipeline
import os
import numpy as np
import kiui
import cv2
from basicsr.utils import img2tensor
from tqdm import tqdm
import torch.nn as nn
import json
from torchvision.utils import save_image


def get_camera_view(camera_path, i):
    """Load one of the default cameras for the scene."""
    cam_path = os.path.join(camera_path, "camera.json")
    with open(cam_path) as f:
        data = json.load(f)
        elevation = data[i]['elevation']
        azimuth = data[i]['azimuth']

    return elevation, azimuth


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


def create_circle_mask(min_x, min_y, max_x, max_y, scale, shape):
    # Calculate the center point of the circle
    center_x = (min_x//scale + max_x//scale) / 2
    center_y = (min_y//scale + max_y//scale) / 2

    # Calculate the radius of the circle as half the distance between min and max coordinates
    radius = max(max_x - min_x, max_y - min_y) / 2 + 10
    if radius < 11:
        radius = 0
    # Create an array of zeros with the specified shape
    new_array = np.zeros(shape, dtype=np.uint8)
    # Create a meshgrid of points corresponding to the shape of the new array
    y, x = np.ogrid[:shape[2], :shape[3]]
    # Calculate the distance of each point from the center of the circle
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Set values in the circular region to 1
    new_array[:,:,distances < radius] = 1

    return new_array


def create_circle_mask_from_lines(lines, shape, scale):
    # Create an array of zeros with the specified shape
    mask = np.zeros(shape, dtype=np.uint8)

    for i in range(0, len(lines), 2):
        # Extract points defining the line
        x1, y1 = lines[i]
        x2, y2 = lines[i+1]
        
        # Calculate the center of the line
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate the radius as half the distance between the two points
        radius = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2 * scale
        
        # Create a meshgrid of points corresponding to the shape of the mask
        y, x = np.ogrid[:shape[2], :shape[3]]
        
        # Calculate the distance of each point from the center of the circle
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Set values in the circular region to 1
        mask[:,:,distances < radius] = 1
    
    return mask


def save_np2img(mask_i, filename):
    mask_i_np = mask_i
    # Transpose the dimensions to match the format expected by PIL (H, W, C)
    mask_i_np = np.transpose(mask_i_np, (1, 2, 0))
    # Normalize the image to the range [0, 255] and convert to uint8
    mask_i_np = (mask_i_np - mask_i_np.min()) / (mask_i_np.max() - mask_i_np.min()) * 255
    mask_i_np = mask_i_np.astype(np.uint8)
    # Convert the NumPy array to a PIL image
    mask_i_img = Image.fromarray(mask_i_np)
    # Save the image
    # filename = f'mask_i_image_{i}.png'
    mask_i_img.save(filename)


def pre_process_drag(masks, 
                     drag_points, 
                     scale, 
                     input_scale, 
                     up_scale, 
                     up_ft_index, 
                     w_edit,
                     w_inpaint, 
                     w_content, 
                     precision, 
                     latent_in, 
                     device):  #[4, 4, 32, 32]
    dict_masks = []
    mask_x0s = []
    mask_tars = []
    mask_curs = []
    mask_others = []

    for i in range(latent_in.shape[0]):
        # import pdb;pdb.set_trace()
        mask_i = masks[i,:,:,:]

        drag_point_i = drag_points[i] # original scale in [512,512]
        drag_point_i = drag_point_i*2
        
        mask_x0 = torch.from_numpy(mask_i)[0] #[256, 256])
        dict_mask = {}
        dict_mask['base'] = mask_x0
        mask_x0 = (mask_x0>0.5).float().to(device=device, dtype=precision)
        mask_other = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]), int(mask_x0.shape[-1])))<0.5 #[1, 1, 256, 256]

        mask_tar = []
        mask_cur = []
        x=[]
        y=[]
        x_cur = []
        y_cur = []
        # import pdb;pdb.set_trace()
        for idx, point in enumerate(drag_point_i):
            if idx%2 == 0:
                y.append(point[1]*input_scale)
                x.append(point[0]*input_scale)
            else:
                y_cur.append(point[1]*input_scale)
                x_cur.append(point[0]*input_scale)
        for p_idx in range(len(x)):
            mask_tar_i = torch.zeros(int(mask_x0.shape[-2]), int(mask_x0.shape[-1])).to(device=device, dtype=precision) #[256, 256]
            mask_cur_i = torch.zeros(int(mask_x0.shape[-2]), int(mask_x0.shape[-1])).to(device=device, dtype=precision) #[256, 256]
            y_tar_clip = int(np.clip(y[p_idx]//scale, 1, mask_tar_i.shape[0]-2))
            x_tar_clip = int(np.clip(x[p_idx]//scale, 1, mask_tar_i.shape[0]-2))
            y_cur_clip = int(np.clip(y_cur[p_idx]//scale, 1, mask_cur_i.shape[0]-2))
            x_cur_clip = int(np.clip(x_cur[p_idx]//scale, 1, mask_cur_i.shape[0]-2))
            # Ensure the coordinates are within the valid range
            y_tar_clip = max(2, min(y_tar_clip, mask_tar_i.shape[0] - 2))
            x_tar_clip = max(2, min(x_tar_clip, mask_tar_i.shape[1] - 2))
            y_cur_clip = max(2, min(y_cur_clip, mask_cur_i.shape[0] - 2))
            x_cur_clip = max(2, min(x_cur_clip, mask_cur_i.shape[1] - 2))
            mask_tar_i[y_tar_clip-2:y_tar_clip+2,x_tar_clip-2:x_tar_clip+2]=1
            mask_cur_i[y_cur_clip-2:y_cur_clip+2,x_cur_clip-2:x_cur_clip+2]=1
            mask_tar_i = mask_tar_i>0.5
            mask_cur_i = mask_cur_i>0.5
            mask_tar.append(mask_tar_i)
            mask_cur.append(mask_cur_i)

            patch_size = 4  # 6
            half_patch_size = patch_size // 2

             # Function to get valid patch bounds
            def get_patch_bounds(center, half_size, max_size):
                start = max(center - half_size, 0)
                end = min(center + half_size, max_size)
                return start, end

            # Get bounds for the current patch
            y_cur_start, y_cur_end = get_patch_bounds(y_cur_clip // 8, half_patch_size, latent_in.shape[2])
            x_cur_start, x_cur_end = get_patch_bounds(x_cur_clip // 8, half_patch_size, latent_in.shape[3])

            y_tar_start, y_tar_end = get_patch_bounds(y_tar_clip // 8, half_patch_size, latent_in.shape[2])
            x_tar_start, x_tar_end = get_patch_bounds(x_tar_clip // 8, half_patch_size, latent_in.shape[3])

            patch_height = min(y_cur_end - y_cur_start, y_tar_end - y_tar_start)
            patch_width = min(x_cur_end - x_cur_start, x_tar_end - x_tar_start)

            y_cur_end = y_cur_start + patch_height
            x_cur_end = x_cur_start + patch_width
            y_tar_end = y_tar_start + patch_height
            x_tar_end = x_tar_start + patch_width

            latent_in[i, :, y_cur_start:y_cur_end, x_cur_start:x_cur_end] = latent_in[i, :, y_tar_start:y_tar_end, x_tar_start:x_tar_end]
        
        dict_masks.append(dict_mask)
        mask_x0s.append(mask_x0)
        mask_tars.append(mask_tar)
        mask_curs.append(mask_cur)
        mask_others.append(mask_other)

    return {
        "dict_masks":dict_masks,
        "mask_x0s":mask_x0s,
        "mask_tars":mask_tars,
        "mask_curs":mask_curs,
        "mask_others":mask_others,
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "w_edit": w_edit,
        "w_inpaint": w_inpaint,
        "w_content": w_content,
        "latent_in":latent_in,
    }


def guidance_drag(
    pipe,
    mask_x0s,
    mask_curs, 
    mask_tars, 
    mask_others, 
    latent, 
    latent_noise_ref,  #ddim_latents
    latent_ori,
    t, 
    up_ft_index, 
    up_scale, 
    prompt_embeds_neg,
    text_embeddings,
    c2ws,
    energy_scale,
    w_edit,
    w_inpaint,
    w_content,
    device,
    dict_masks = None,
):
    cos = nn.CosineSimilarity(dim=1)
    actual_num_frames = len(mask_x0s)
    up_ft_index = [0, 1, 2, 3]
    used_features = 0

    with torch.no_grad():
        up_ft_tar = pipe.get_unet_features(latent_noise_ref, t, guidance_scale=1.0, actual_num_frames=4, prompt_embeds_pos=text_embeddings, c2ws=c2ws, device=device, return_unet_feature=True) 
        for f_id in range(len(up_ft_index)):
            f_id += used_features
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (up_ft_tar[-1].shape[-2]*up_scale*4, up_ft_tar[-1].shape[-1]*up_scale*4)) #[4, X, 256, 256]

    latent = latent.detach().requires_grad_(True)
    up_ft_cur = pipe.get_unet_features(latent, t, guidance_scale=1.0, actual_num_frames=4, prompt_embeds_pos=text_embeddings, c2ws=c2ws, device=device, return_unet_feature=True) #[4, 4, 32, 32] add camera condition
    for f_id in range(len(up_ft_index)):
        f_id += used_features
        up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale*4, up_ft_cur[-1].shape[-1]*up_scale*4))#[4, X, 256, 256]

    # moving loss
    loss_edit = 0
    loss_con = 0
    for i in range(actual_num_frames):
        mask_cur = mask_curs[i]
        mask_tar = mask_tars[i]
        mask_x0 = mask_x0s[i]
        mask_other = mask_others[i]
        for f_id in range(len(up_ft_index)):
            f_id += used_features
            up_ft_cur_f_id = up_ft_cur[f_id][i].unsqueeze(0)
            up_ft_tar_f_id = up_ft_tar[f_id][i].unsqueeze(0)
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar): #[256, 256] [256, 256]
                up_ft_cur_vec = up_ft_cur_f_id[mask_cur_i.repeat(1,up_ft_cur_f_id.shape[1],1,1)].view(up_ft_cur_f_id.shape[1], -1).permute(1,0)  
                up_ft_tar_vec = up_ft_tar_f_id[mask_tar_i.repeat(1,up_ft_tar_f_id.shape[1],1,1)].view(up_ft_tar_f_id.shape[1], -1).permute(1,0) 
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
                loss_edit = loss_edit + w_edit/(1+4*sim.mean())

                mask_overlap = ((mask_cur_i.float()+mask_tar_i.float())>1.5).float()
                mask_non_overlap = (mask_tar_i.float()-mask_overlap)>0.5
                # import pdb;pdb.set_trace()
                up_ft_cur_non_overlap = up_ft_cur_f_id[mask_non_overlap.repeat(1,up_ft_cur_f_id.shape[1],1,1)].view(up_ft_cur_f_id.shape[1], -1).permute(1,0)
                up_ft_tar_non_overlap = up_ft_tar_f_id[mask_non_overlap.repeat(1,up_ft_tar_f_id.shape[1],1,1)].view(up_ft_tar_f_id.shape[1], -1).permute(1,0)
                sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
                loss_edit = loss_edit + w_inpaint*sim_non_overlap.mean()
        # consistency loss
        for f_id in range(len(up_ft_index)):
            f_id += used_features
            up_ft_cur_f_id = up_ft_cur[f_id][i].unsqueeze(0)
            up_ft_tar_f_id = up_ft_tar[f_id][i].unsqueeze(0)
            sim_other = (cos(up_ft_tar_f_id, up_ft_cur_f_id)[0][mask_other[0,0]]+1.)/2.
            loss_con = loss_con+w_content/(1+4*sim_other.mean())
        loss_edit += loss_edit/len(up_ft_cur)/len(mask_cur)
        loss_con += loss_con/len(up_ft_cur)

    cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0] #[4, 4, 64, 64])
    cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0] #[4, 4, 64, 64])
    
    guidance = cond_grad_edit
    for i in range(actual_num_frames):
        mask_x0 = mask_x0s[i]
        mask = F.interpolate(mask_x0[None,None], (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance[i,...] = cond_grad_edit[i,...].detach()*4e-2*mask + cond_grad_con[i,...].detach()*4e-2*(1-mask)
    pipe.unet.zero_grad()
    return guidance


def execute_drag(
    pipe,
    edit_kwargs,
    latent: Optional[torch.FloatTensor] = None, #pre-processed latents:[4, 4, 64, 64]
    prompt = None,
    guidance_scale: Optional[float] = 7.5,
    num_inference_steps: int = 100,
    latent_noise_ref = None, #len(latent_noise_ref)=51 --- ddim latent
    start_time=100,
    energy_scale = 0,
    SDE_strength = 0.4,
    SDE_strength_un = 0,
    c2ws = None,
    latent_ori: Optional[torch.FloatTensor] = None, #pre-processed latents:[4, 4, 64, 64] latent_in_ddim_ori,
    alg='D+',
    use_text_opt=False,
    text_embeddings_com:  Optional[torch.FloatTensor] = None,
    num_steps = 100,
    device = torch.device("cuda:0"),
    mvdream_path = None,
    ):
    num_inference_steps = num_steps
    start_time = num_steps
    print('Start Editing:')
    if use_text_opt:
        _prompt_embeds = pipe._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1*4,
                do_classifier_free_guidance=guidance_scale,
                negative_prompt="",
            )  # type: ignore
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2) # torch.Size([4, 77, 1024]), torch.Size([4, 77, 1024])
        prompt_embeds_pos = torch.cat([pipe.text_embeddings] * 4)
    else:
        _prompt_embeds = pipe._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1*4,
                do_classifier_free_guidance=guidance_scale,
                negative_prompt="",
            )  # type: ignore
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2) # torch.Size([4, 77, 1024]), torch.Size([4, 77, 1024])
    
    if text_embeddings_com is not None:
        prompt_embeds_pos = torch.cat([text_embeddings_com] * 4)

    # 2. set schedular
    pipe.scheduler = DDIMScheduler.from_pretrained(mvdream_path, subfolder='scheduler')
    pipe.scheduler.set_timesteps(num_inference_steps) 

    # 3. compute gudance
    for i, t in enumerate(tqdm(pipe.scheduler.timesteps[-start_time:])):
        next_timestep = min(t - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps, 999)
        next_timestep = max(next_timestep, 0)
        if energy_scale==0 or alg=='D':
            repeat=1
        elif num_steps*0.4<i<num_steps*0.6 and i%2==0 : 
            repeat = 3
        else:
            repeat = 1
        stack = []
        # repeat = 3
        for ri in range(repeat):
            # latent_in = torch.cat([latent] * 2) #[8, 4, 32, 32] 
            latent_in = latent
            with torch.no_grad():
                noise_pred = pipe.apply_unet(latent_in, t, guidance_scale, 4, prompt_embeds_neg, prompt_embeds_pos, c2ws, device=device) #[4, 4, 32, 32] add camera condition
            if energy_scale!=0 and i<num_steps*0.8 and (alg=='D' or i%2==0 or i<10):
                # editing guidance
                noise_pred_org = noise_pred
                guidance = guidance_drag(pipe, latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], latent_ori = latent_ori, t=t, 
                                         prompt_embeds_neg=prompt_embeds_neg, text_embeddings=prompt_embeds_pos, 
                                         c2ws=c2ws, energy_scale=energy_scale, device=device, **edit_kwargs)
                noise_pred = noise_pred + guidance
            else:
                noise_pred_org=None

            # zt->zt-1
            prev_timestep = t - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
            alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else pipe.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            if 0.2*num_steps<i<0.4*num_steps:
                eta, eta_rd = SDE_strength_un, SDE_strength
            else:
                eta, eta_rd = 0., 0.

            variance = pipe.scheduler._get_variance(t, prev_timestep)
            std_dev_t = eta * variance ** (0.5)
            std_dev_t_rd = eta_rd * variance ** (0.5)
            if noise_pred_org is not None:
                pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
            else:
                pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

            latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd

            # Regional SDE
            if True: #(eta_rd > 0 or eta>0) and alg=='D+':
                variance_noise = torch.randn_like(latent_prev)
                variance_rd = std_dev_t_rd * variance_noise
                variance = std_dev_t * variance_noise
                for jj in range(4):
                    mask_x0i = edit_kwargs["mask_x0s"][jj]
                    mask = F.interpolate(mask_x0i[None,None], (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]))
                    mask = (mask>0).to(dtype=latent.dtype)
                    latent_prev[jj,...] = (latent_prev[jj,...]+variance[jj,...])*(1-mask) + (latent_prev_rd[jj,...]+variance_rd[jj,...])*mask
            
            if repeat>1:
                with torch.no_grad():
                    alpha_prod_t = pipe.scheduler.alphas_cumprod[next_timestep]
                    alpha_prod_t_next = pipe.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t

                    model_output = pipe.apply_unet(latent_prev, next_timestep, 1.0, 4, 
                                    prompt_embeds_neg, prompt_embeds_pos, c2ws, device=device) #[4, 4, 32, 32] add camera condition
                    next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                    latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        
        latent = latent_prev #[4, 4, 64, 64]

    return latent      


def mvdrag(opt: dict = None,
           device: torch.device = torch.device("cuda:0"),
           dtype: torch.dtype = torch.float32,
           workspace_name: str = None,
           verify: bool = True) -> torch.Tensor:
    """
    Main function for multi-view drag pipeline.
    Args:
        opt: configuration dictionary
        device: torch device
        dtype: torch dtype
        workspace_name: output directory
        verify: whether to verify DDIM latent
    Returns:
        Reconstructed images (tensor)
    """
    # Load pipeline and VAE
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(opt.mvdream_path, subfolder='scheduler')
    pipe = MVDreamPipeline.from_pretrained(
        opt.mvdream_path,
        scheduler=inverse_scheduler,
        safety_checker=None,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    pipe.to(device)
    vae = pipe.vae

    # Load images and dragging points
    subimages = load_and_split_image(opt.image_path)
    drag_points, drag_points_all = load_drag_points(opt.drag_points_path_occ, opt.drag_points_path_all)

    # Create masks
    masks = create_masks(drag_points, opt, workspace_name)

    # Encode images to latents
    latents = encode_to_latents(subimages, vae, device)

    # Set cameras
    cameras = get_camera(4, 0, 0).to(device)

    # DDIM inversion
    all_ddim_latents = pipe.invert(
        prompt=opt.prompt,
        negative_prompt="",
        guidance_scale=1.,
        c2ws=cameras,
        width=256,
        height=256,
        output_type='latent',
        num_inference_steps=opt.num_steps,
        device=device,
        latents=latents
    )

    # Pre-process for drag
    latent_in_ddim = all_ddim_latents[-1]
    latent_in_ddim_ori = latent_in_ddim.clone()
    edit_kwargs = pre_process_drag(
        latent_in=latent_in_ddim,
        masks=masks,
        drag_points=drag_points,
        scale=2,
        input_scale=1,
        up_scale=2,
        up_ft_index=[1, 2],
        w_edit=opt.w_edit,
        w_content=opt.w_content,
        w_inpaint=0.1,
        precision=dtype,
        device=device,
    )

    # Execute drag
    latent_in = edit_kwargs.pop('latent_in')
    latent_rec = execute_drag(
        pipe,
        latent=latent_in,
        prompt=opt.prompt_edit,
        guidance_scale=opt.guidance_scale,
        energy_scale=55,
        latent_noise_ref=all_ddim_latents,
        SDE_strength=opt.SDE_strength,
        c2ws=cameras,
        latent_ori=latent_in_ddim_ori,
        use_text_opt=False,
        num_steps=opt.num_steps,
        device=device,
        mvdream_path=opt.mvdream_path,
        edit_kwargs=edit_kwargs,
    )

    # Decode reconstructed images
    img_rec = pipe.decode_latents(latent_rec)
    torch.cuda.empty_cache()

    # Optional: verify DDIM latent
    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained(opt.mvdream_path, subfolder='scheduler')
        images = pipe(
            prompt=opt.prompt,
            c2ws=cameras,
            negative_prompt="",
            guidance_scale=1.0,
            num_inference_steps=opt.num_steps,
            latents=latent_in_ddim_ori,
            output_type="np"
        ).images[0]
        grid = np.concatenate([
            np.concatenate([images[0], images[2]], axis=0),
            np.concatenate([images[1], images[3]], axis=0),
        ], axis=1)
        kiui.write_image(f'drag_sample_inversion.jpg', grid)

    return img_rec


if __name__ == '__main__':
    mvdrag(num_steps=150, verify=True)

# ===== Utility functions for mvdrag main pipeline =====
def load_and_split_image(image_path):
    """Load image and split into 4 subimages."""
    image = Image.open(image_path)
    image_np = np.array(image).astype(np.float32) / 255.0
    h, w = image_np.shape[:2]
    subimages = [
        image_np[0:h//2, 0:w//2],
        image_np[0:h//2, w//2:w],
        image_np[h//2:h, 0:w//2],
        image_np[h//2:h, w//2:w]
    ]
    return subimages

def load_drag_points(points_path, points_all_path):
    """Load dragging points from files."""
    points = np.loadtxt(points_path)
    points_all = np.loadtxt(points_all_path)
    drag_points = [points[2*i:2*i+2, :].T.astype(np.int32) for i in range(4)]
    drag_points_all = [points_all[2*i:2*i+2, :].T.astype(np.int32) for i in range(4)]
    return drag_points, drag_points_all

def create_masks(drag_points, opt, workspace_name):
    """Create masks based on dragging points."""
    masks = []
    mask0 = np.zeros((1, 3, 256, 256), dtype=np.uint8)
    scale = opt.scale
    for i in range(4):
        data_i = drag_points[i]
        mask = create_circle_mask_from_lines(data_i, mask0.shape, scale)
        masks.append(mask)
        save_np2img(mask[0, ...], os.path.join(workspace_name, f"mask_{i}.png"))
    masks = np.stack(masks, axis=0)
    masks = masks.squeeze(1)
    return masks

def encode_to_latents(subimages, vae, device):
    """Encode subimages to latents using VAE."""
    subimages_tensors = [torch.from_numpy(subimg).permute(2, 0, 1) for subimg in subimages]
    latents = []
    for subimg_tensor in subimages_tensors:
        subimg_tensor = subimg_tensor.unsqueeze(0).to(device)
        latent = img_to_latents(subimg_tensor, vae)
        latents.append(latent)
    latents = torch.stack(latents, axis=0)
    latents = latents.squeeze(1)
    return latents