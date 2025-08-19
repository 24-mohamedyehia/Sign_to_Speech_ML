# V1_sign_language
This repository contains the code for the V1_sign_language project, which is designed to recognize and translate sign language gestures into text.

## 📋 Requirements
- Python 3.11

## 🚀 Quickstart

1- Clone repo
 - `git clone https://github.com/24-mohamedyehia/Sign_to_Speech_ML`

2- 📦 Install Python Using Miniconda
 - Download and install MiniConda from [here](https://www.anaconda.com/docs/getting-started/miniconda/main#quick-command-line-install)

3- Create a new environment using the following command:
```bash
$ conda create --name sign_language python=3.11 -y
```

4- Activate the environment:
```bash
$ conda activate sign_language 
```

5- Install the required packages
```bash
$ pip install -r requirements.txt
```

## Testing the Model

1. Real Time sign to text
   - To test the real-time sign to text functionality, run the following script:
   ```bash
   $ python real_time_sign_to_text.py
   ```

2. Real time sign to speech
   - To test the real-time sign to speech functionality, run the following script:
   ```bash
   $ python real_time_sign_to_speech.py
   ```

3. Test sign to text with video file
   - To test the sign to text functionality with a video file, run the following script:
   Run => test_sign_to_text_with_video_file.ipynb 