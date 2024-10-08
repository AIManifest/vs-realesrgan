import math
import os
import warnings
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm.auto import tqdm

from .rrdbnet_arch import RRDBNet
from .srvgg_arch import SRVGGNetCompact
from .ESRGAN import RRDBNet as ESRGAN

import time
import skvideo.io
import _thread
from queue import Queue, Empty
from PIL import Image
from spandrel import ModelLoader
import spandrel_extra_arches
# from .SPAN import SPAN
# from spandrel.architectures.SPAN import SPAN
from .spandrel_SPAN import SPAN
from spandrel.util import KeyCondition, get_scale_and_output_channels

__version__ = "5.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


class RealESRGANModel(IntEnum):
    ESRGAN_SRx4 = 0
    RealESRGAN_x2plus = 1
    RealESRGAN_x4plus = 2
    RealESRGAN_x4plus_anime_6B = 3
    realesr_animevideov3 = 4
    realesr_general_x4v3 = 5

    AnimeJaNai_HD_V3_Compact_2x = 100
    AnimeJaNai_HD_V3_UltraCompact_2x = 101
    AnimeJaNai_HD_V3_SuperUltraCompact_2x = 102

    AniScale2_Compact_2x = 200
    AniScale2_Refiner_1x = 201
    OpenProteus_Compact_2x = 202
    Ani4Kv2_Compact_2x = 203
    Ani4Kv2_UltraCompact_2x = 204
    ESRGAN_4x_UltraMix_Balanced = 205

def numpy_to_pil_display(np_array):
    """
    Convert a NumPy array to a PIL Image and display it.
    
    Parameters:
    np_array (numpy.ndarray): The NumPy array to be converted.
    
    Returns:
    PIL.Image.Image: The converted PIL Image.
    """
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(np_array.astype('uint8'))

    display(pil_image)

    return pil_image


@torch.inference_mode()
def realesrgan(
        clip_path: str,
        output_path: str,
        device_index: int = 0,
        num_streams: int = 1,
        model: RealESRGANModel = RealESRGANModel.realesr_general_x4v3,
        model_path: str | None = None,
        denoise_strength: float = 0.5,
        tile: list[int] = [0, 0],
        tile_pad: int = 8,
        trt: bool = False,
        trt_debug: bool = False,
        trt_workspace_size: int = 0,
        trt_int8: bool = False,
        trt_int8_sample_step: int = 72,
        trt_int8_batch_size: int = 1,
        trt_cache_dir: str = model_dir,
        use_fp16: bool = False,
        custom_model: bool = False
    ) -> None:
    torch.set_float32_matmul_precision("high")

    # Assert Settings
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"Input video file '{clip_path}' not found")

    if not torch.cuda.is_available():
        raise RuntimeError("realesrgan: CUDA is not available")

    if num_streams < 1:
        raise ValueError("realesrgan: num_streams must be at least 1")

    # if model not in RealESRGANModel:
    #     raise ValueError("realesrgan: model must be one of the members in RealESRGANModel")

    if denoise_strength < 0 or denoise_strength > 1:
        raise ValueError("realesrgan: denoise_strength must be between 0.0 and 1.0 (inclusive)")

    if not isinstance(tile, list) or len(tile) != 2:
        raise ValueError("realesrgan: tile must be a list with 2 items")

    if trt and trt_int8:
        raise ValueError("realesrgan: INT8 mode not implemented in this conversion")

    #Capture Video Information
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file '{clip_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Set Dtype
    fp16 = use_fp16
    if fp16:
        print("***Overriding tensors to fp16***")
        
    dtype = torch.half if fp16 else torch.float

    #Set Device
    device = torch.device("cuda", device_index)

    #Set Streams
    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    #Set Model Path
    if model_path is None:
        print(f'Using {model} for inference')
        if model == RealESRGANModel.ESRGAN_4x_UltraMix_Balanced:
            model = "4x-UltraMix_Balanced"
            model_name = f"{model}.pth"
        else:
            model_name = f"{RealESRGANModel(model).name}.pth"

        model_path = os.path.join(model_dir, model_name)
        print(f"Using {model_path}")
    else:
        print(f'Using {model_path} for inference')
        model_path = os.path.realpath(model_path)
        model_name = os.path.basename(model_path)
        print(f'Using {model_path} for inference')
        print(f'Using {model_name} for inference')

    #Load State Dict
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    #Model Set Up
    if not custom_model:    
        if model == RealESRGANModel.realesr_general_x4v3 and denoise_strength != 1:
            wdn_model_path = model_path.replace("realesr_general_x4v3", "realesr_general_wdn_x4v3")
            dni_weight = [denoise_strength, 1 - denoise_strength]
    
            net_b = torch.load(wdn_model_path, map_location=device, weights_only=True)
            if "params_ema" in net_b:
                net_b = net_b["params_ema"]
            elif "params" in net_b:
                net_b = net_b["params"]
    
            for k, v in state_dict.items():
                state_dict[k] = dni_weight[0] * v + dni_weight[1] * net_b[k]
    
        if "conv_first.weight" in state_dict:
            num_feat = state_dict["conv_first.weight"].shape[0]
            num_block = int(list(state_dict)[-11].split(".")[1]) + 1
            num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]
    
            match state_dict["conv_first.weight"].shape[1]:
                case 48:
                    scale = 1
                case 12:
                    scale = 2
                case _:
                    scale = 4
    
            module = RRDBNet(3, 3, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch, scale=scale)
        else:
            num_feat = state_dict["body.0.weight"].shape[0]
            num_conv = int(list(state_dict)[-1].split(".")[1]) // 2 - 1
            scale = math.isqrt(state_dict[list(state_dict)[-1]].shape[0] // 3)
    
    
            module = SRVGGNetCompact(3, 3, num_feat=num_feat, num_conv=num_conv, upscale=scale, act_type="prelu")
        
    elif custom_model:
        print("ESRGAN/SPAN")
        if "UltraMix" in model_path or "model.0.weight" in state_dict:
            print("Loading UltraMix")
            module = ESRGAN(state_dict)
            scale = 4
        elif "ClearRealityV1" in model_path or "conv_1.sk.weight" in state_dict:
            print("Loading ClearReality")
            # module = ModelLoader().load_from_file(model_path)
            # module = span.load_state_dict(state_dict).eval().cuda()
            # module = SPAN(3, 3, upscale=4, feature_channels=48)
            num_in_ch = 3
            num_out_ch = 3
            feature_channels = 48
            upscale = 4
            bias = True  # unused internally
            norm = True
            img_range = 255.0  # cannot be deduced from state_dict
            rgb_mean = (0.4488, 0.4371, 0.4040)  # cannot be deduced from state_dict
    
            num_in_ch = state_dict["conv_1.sk.weight"].shape[1]
            feature_channels = state_dict["conv_1.sk.weight"].shape[0]
    
            # pixelshuffel shenanigans
            upscale, num_out_ch = get_scale_and_output_channels(
                state_dict["upsampler.0.weight"].shape[0],
                num_in_ch,
            )
    
            # norm
            if "no_norm" in state_dict:
                norm = False
                state_dict["no_norm"] = torch.zeros(1)
    
            module = SPAN(
                num_in_ch=num_in_ch,
                num_out_ch=num_out_ch,
                feature_channels=feature_channels,
                upscale=upscale,
                bias=bias,
                norm=norm,
                img_range=img_range,
                rgb_mean=rgb_mean,
            )
            # module.float().cuda().eval()
            scale = 4
        if "conv_first.weight" in state_dict:
            print("Using Spandrel, turning off TRT")
            module = ModelLoader().load_from_file(model_path)
            scale = 1
            trt=False

    # model_name = os.path.basename(model_path)
    # model_name = os.path.splitext(model_name)[0]

    #Set Model Dtype
    if fp16:
        module.half()

    #Set Padding Vars
    match scale:
        case 1:
            modulo = 4
        case 2:
            modulo = 2
        case _:
            modulo = 1

    if all([t > 0 for t in tile]):
        pad_w = math.ceil(min(tile[0] + 2 * tile_pad, width) / modulo) * modulo
        pad_h = math.ceil(min(tile[1] + 2 * tile_pad, height) / modulo) * modulo
    else:
        pad_w = math.ceil(width / modulo) * modulo
        pad_h = math.ceil(height / modulo) * modulo

    #Load State Dict to Model
    module.load_state_dict(state_dict, strict=True)

    # #ESRGAN, 
    if model == RealESRGANModel.ESRGAN_4x_UltraMix_Balanced:
        print('removing grad')
        for k, v in module.named_parameters():
            v.requires_grad = False

    # #Initiate Model to Eval and Send to Cuda
    module = module.cuda().eval()

    tmp_module = module

    use_denoiser_model = False
    if use_denoiser_model:
        spandrel_extra_arches.install()

        denoise_model = ModelLoader().load_from_file(r"/workspace/1x_ArtClarity.pth")
        if fp16:
            denoise_model.half()
        denoise_model.eval().cuda()
        print("Successfully Loaded Denoiser Model")

    #TRT Conversion
    if trt:
        import torch_tensorrt
        from torch_tensorrt import Input
        import tensorrt
        
        dtype = torch.float16 if fp16 else torch.float
        # module = module.to(device=device, dtype=dtype).eval()
        trt_engine_path = os.path.join(
                os.path.realpath(trt_cache_dir),
                (
                    f"{model_name}"
                    + f"_{'trt' if trt else ''}"
                    + f"_{'custom_model' if custom_model else ''}"
                    + f"_{pad_w}x{pad_h}"
                    + f"_{'int8' if trt_int8 else 'fp16' if fp16 else 'fp32'}"
                    + f"_{torch.cuda.get_device_name(device)}"
                    + f"_trt-{tensorrt.__version__}"
                    + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                    + ".ts"
                ),
            )

        print(f'trt engine path --> {trt_engine_path}')
        if not os.path.isfile(trt_engine_path):
            print('initiating model conversion to trt')
            inputs = [torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device)]

            traced_module = torch.jit.trace(module, inputs)

            if trt_int8:
                dataset = torch_tensorrt.ptq.DataLoaderCalibrator(
                    [Input((1, 3, height, width))],
                    cache_file=os.path.splitext(trt_engine_path)[0] + ".calib.cache",
                    use_cache=True,
                    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                )

                with torch_tensorrt.logging_errors(is_py_trt_exception=trt_debug):
                    module = torch_tensorrt.compile(
                        module,
                        inputs=inputs,
                        enabled_precisions={torch.int8},
                        truncate_long_and_double=True,
                        max_batch_size=trt_int8_batch_size,
                        calib_data_loader=dataset,
                        workspace_size=trt_workspace_size,
                    )
            else:
                compiled_module = torch_tensorrt.compile(
                    traced_module,
                    ir="ts",
                    inputs=inputs,
                    enabled_precisions={torch.float16 if fp16 else torch.float},
                    workspace_size=trt_workspace_size,
                    truncate_long_and_double=True
                )

            scripted_module = torch.jit.script(compiled_module)
            scripted_module.save(trt_engine_path)

        loaded_module = torch.jit.load(trt_engine_path)
        backend = Backend.TensorRT(module=module)
        module = loaded_module
        print("Using Loaded Module")
    else:
        backend = Backend.Eager(module=module)

    #Output Video Settings
    clip_name = os.path.splitext(clip_path)[0]
    clip_dir = os.path.dirname(clip_path)
    name_length = len([f for f in os.listdir(clip_dir) if os.path.isfile(clip_path) and clip_name in f])
    output_path = os.path.join(os.path.dirname(clip_path), clip_name + f"{model_name}_{name_length+1:03d}_output_clip.mp4") if output_path is None else output_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width * scale, height * scale))

    #Create Frame Reader
    videogen = skvideo.io.vreader(clip_path)

    #Create Buffers
    use_png = False
    def clear_write_buffer(use_png, write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if use_png:
                cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                cnt += 1
            else:
                # print('writing frame', item[:, :, ::-1].shape)
                writer.write(item[:, :, ::-1])

    use_montage = False
    def build_read_buffer(use_montage, read_buffer, videogen):
        try:
            for frame in videogen:
                # if not user_args.img is None:
                #     frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                if use_montage:
                    frame = frame[:, left: left + w]
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (use_montage, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (use_png, write_buffer))

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), unit="frame")
    cap.release()

    frame_idx = 0
    start_time = time.time()

    write_png_file = False
    if write_png_file:
        png_dir = os.path.join(os.path.dirname(output_path), "tmpdir")
        os.makedirs(png_dir, exist_ok=True)

    #Inference Loop
    while True:
        frame = read_buffer.get()
        if frame is None:
            print("frame is none")
            break

        frame = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        if fp16:
            frame = frame.half()

        i = frame_idx % num_streams
        stream_lock[i].acquire()

        torch.cuda.current_stream().wait_stream(stream[i])
        with torch.cuda.stream(stream[i]):
            with torch.no_grad():
                output = module(frame)
                if use_denoiser_model:
                    output = denoise_model(output)
            output = output.squeeze().permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy()
            
            if write_png_file: 
                cv2.imwrite(os.path.join(png_dir, f"frame_{frame_idx:09d}.png"), output)
                
        write_buffer.put(output)
        stream_lock[i].release()

        pbar.update(1)
        frame_idx += 1

    write_buffer.put(None)
        
    while(not write_buffer.empty()):
        time.sleep(0.1)

    pbar.close()
    writer.release()

    print(f"Video Upscaling completed in {time.time()-start_time:.2f}")

    return output_path
