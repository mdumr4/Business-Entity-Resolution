# GCP Deep Learning VM - L4 GPU Instance Creation Script
gcloud compute instances create amazonml-gpu-vm `
    --project=amazonml-509705 `
    --zone=us-central1-a `
    --machine-type=g2-standard-8 `
    --maintenance-policy=TERMINATE `
    --accelerator=type=nvidia-l4,count=1 `
    --image-family=common-cu121-debian-11 `
    --image-project=deeplearning-platform-release `
    --boot-disk-size=250GB `
    --metadata="install-nvidia-driver=True"
