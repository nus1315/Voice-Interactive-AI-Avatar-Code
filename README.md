# Voice-Interactive 2D Avatar Pipeline Code

This repository contains the automation pipeline and model adapters for evaluating and running four Audio-Driven Talking-Head models in a unified batch process.

## 🚀 Supported Models (Official Repositories)

The pipeline integrates the following state-of-the-art models. Click on the folder names below to visit their official GitHub repositories:

* 📁 [**SadTalker**](https://github.com/OpenTalker/SadTalker) - Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation
* 📁 [**EchoMimic**](https://github.com/AntGroup/EchoMimic) - Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditions
* 📁 [**Ditto-Talkinghead**](https://github.com/antgroup/ditto-talkinghead) - Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis
* 📁 [**IMTalker**](https://github.com/cbsjtu01/IMTalker) - Efficient Audio-Driven Talking Face Generation with Implicit Motion Transfer

## 🛠️ Usage

This repository provides two main scripts:

1. **`run_batch.sh`**: A shell script to batch process multiple audio files against a single source image across all models.
2. **`adapters.py`**: The unified interface classes that handle model-specific configurations, environments, and output locations.

To run the batch pipeline, simply configure your `IMAGE` and `AUDIOS` variables inside `run_batch.sh` and execute:

```bash
bash run_batch.sh
```

## 📝 Notes

Ensure that you have properly cloned each model's repository and downloaded their respective checkpoints before attempting to run the batch script. The `adapters.py` script automatically configures the environments needed to interface with them seamlessly.
