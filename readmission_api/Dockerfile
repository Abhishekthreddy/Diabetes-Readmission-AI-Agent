FROM public.ecr.aws/lambda/python:3.12

# Amazon Linux 2023 uses dnf (not yum or apt)
RUN dnf install -y gcc gcc-c++ make cmake && \
    dnf clean all

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY api/ ./api/
COPY ml/models/ ./ml/models/

# Set Lambda handler
CMD ["api.main.handler"]