# FROM isaac_ros_dev-aarch64
# FROM ros:humble
FROM robocaster/flexivpy:latest

ENV DEBIAN_FRONTEND=noninteractive
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
SHELL ["/bin/bash", "-c"]

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]