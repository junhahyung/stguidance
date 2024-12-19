# Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling

## Paper
- Arxiv: [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664)

## Project Page
- [STG Project Page](https://junhahyung.github.io/STGuidance)

## Video Examples
Below are example videos showcasing the enhanced video quality achieved through STG:

### HunyuanVideo


https://github.com/user-attachments/assets/8d98c1ff-4e34-467c-98f4-f4825715b5d0


https://github.com/user-attachments/assets/0a165143-cea8-4936-98e0-b6069129a3e0




### Mochi


https://github.com/user-attachments/assets/b576ca62-b058-4ef7-9bf9-13c3d64f0da0


https://github.com/user-attachments/assets/7e22c92d-0ea9-425c-a090-ce6bae07a71c


### CogVideoX


https://github.com/user-attachments/assets/adc5af40-e50d-4b00-b98b-8e88ee04bae8


https://github.com/user-attachments/assets/fcb8a078-58a5-4e62-a55e-662a0b08216b




### LTX-Video


https://github.com/user-attachments/assets/4cd722cd-c6e8-428d-8183-65e5954a930b




## Todos
- Update README
- Update STG with HunyuanVideo, CogVideoX, Open-Sora, SVD

## Acknowledgements
This project is built upon the following works:
- [Mochi](https://github.com/genmoai/mochi?tab=readme-ov-file)
- [LTX-Video](https://github.com/Lightricks/LTX-Video)

## Start Guide
1. **Mochi**
   - For installation and requirements, refer to the [official repository](https://github.com/genmoai/mochi).
     
   - Update `demos/config.py` with your desired settings and simply run:
     ```bash
     python ./demos/cli.py
     ```

2. **LTX-Video**
   - For installation and requirements, refer to the [official repository](https://github.com/Lightricks/LTX-Video).

   **Using CFG (Default Model):**
   ```bash
   python inference.py --ckpt_dir './weights' --prompt "A man ..."
   ```

   **To utilize STG, use the following command:**
   ```bash
   python inference.py --ckpt_dir './weights' --prompt "A man ..." --stg_mode stg-a --stg_scale 1.0 --stg_block_idx 19 --do_rescaling True
   ```
   Key Parameters:
   - **stg_mode**: Choose between stg-a or stg-r.
   - **stg_scale**: Recommended values are ≤2.0.
   - **stg_block_idx**: Specify the block index for applying STG.
   - **do_rescaling**: Set to True to enable rescaling.
