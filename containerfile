FROM python:3.12

RUN pip install torchmetrics==1.7.1 pytorch-lightning==2.5.1.post
RUN pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install -U nemo_toolkit['asr']

COPY demo.py .
