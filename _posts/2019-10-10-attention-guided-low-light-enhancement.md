---
title: Attention-guided Low-light Image Enhancement
description: 
categories: 
- Low-light image enhancement
- Deep Learning
- Attention
tags:
- 
---

# Attention-guided Low-light Image Enhancement

## 1. Introduction

저조도 환경에서 촬영된 영상은 적정 조도 환경에서 촬영된 영상에 비해 가시성이 떨어집니다. 



이 논문의 Contribution은 다음과 같습니다.

- 


## 2. Related work

저조도 영상 개선 분야는 꾸준히 연구 되어오고 있는데 어떤 방법들이 있는지 알아보겠습니다.

### Traditional enhancement methods

전통적인 영상 개선 방법은 Histogram equalization(HE) 기반 방법과 Retinex 이론을 이용한 방법 두 가지로 나눌 수 있습니다. HE 기반 방법은 밝기 값의 통계적 특성을 이용하여 영상의 dynamic range를 확장하거나 영상의 대비(contrast)를 개선하는 방법입니다. 이 방법은 조명(illumination)을 고려하지 않고 영상 전체의 대비를 개선하면서 과도하게 혹은 저조하게 개선하는 현상을 야기할 수 있습니다.
두 번째로는 Retinex 이론을 이용한 방법입니다. 영상은 반사(reflectance) 성분과 조명(illumination) 성분으로 이루어져 있다는 가정에 따라 정의된 이론이 Retinex 이론입니다. 영상을 반사 성분과 조명 성분으로 나누었을 때, 외부 요인인 조명 성분을 잘 조정하면 저조도 영상을 개선할 수 있습니다. 이렇게 하기 위해서는 영상의 조명 성분을 잘 추정 해야하는데, 수학적으로 ill-posed 문제이기 때문에 여러가지 가정에 따라 조명 성분을 추정하는게 일반적입니다. 이 과정에서 hand-crafted 방법들이 적용되며, 파라미터 튜닝에 의존적이게 됩니다. 또, Retinex 이론 기반 방법들은 대부분 노이즈를 고려하지 않기 때문에 저조도를 개선 시키면서 노이즈 또한 증폭시키는 문제점이 있습니다.

### Learning-based enhancement methods

최근 저조도 영상 개선과 같은 low-level 영상 처리에도 end-to-end network, GAN 과 같은 딥러닝 기술이 적용되면서 해당 분야에도 좋은 성과를 보이고 있습니다.
Retinex 이론을 적용한 


### Image denoising methods



## 3. Dataset

Real-world에서 대량의 {저조도 영상, 적정 조도 영상} 쌍 데이터 셋을 구성하는 것은 어려운 일입니다. 기존에는 적정 조도 영상의 밝기 값을 조절하여 저조도 환경을 인위적으로 합성하거나, 카메라의 노출 정도와 ISO 값을 조절하여 데이터 셋을 구성하는 등의 시도가 있었습니다.
특히 노출 정도와 ISO 값을 조정하는 방법을 이용하여 구성한 데이터 셋은 다양한 저조도 환경에 대응하지 못하는 현상(해당 데이터 셋의 테스트 셋에는 좋은 결과를 보였지만 다른 저조도 영상에 적용했을 때 과도하게 개선하여 saturation 되는 등의)이 나타는 문제점이 있었습니다. 이와 유사하게 극-저조도 환경에서 촬영한 raw 영상과 적정 조도에서 촬영한 영상 쌍으로 데이터 셋을 구성하는 사례도 있었는데, 저조도 영상이 raw 데이터이기 때문에 일반적인 저조도 영상 개선에는 사용하기 어려운 단점이 있습니다. 또 이러한 데이터 셋들은 데이터 양이 비교적 적은 문제점도 있었습니다.
이 논문에서는 PASCAL VOC, MS COCO 등의 널리 사용되는 데이터 셋으로 부터 저조도 환경을 합성하는 방법 이용하여 대량의 {저조도 영상, 적정 조도 영상} 쌍의 데이터 셋을 구성하는 방법을 제안합니다. 

### Candidate Image Selection

학습 데이터 셋에 사용되는 영상은 기본적으로 고화질의 적정 조도에서 촬영된 영상이며, 이러한 고화질 영상에 저조도환경을 합성하여 {저조도 영상, 적정 조도 영상} 쌍을 구성합니다. 이 논문에서는 PASCAL VOC, MS COCO 등의 널리 사용되는 데이터 셋에서 고화질의 적정 조도 영상을 선별하기위해 아래 3가지 방법을 이용합니다.

- Darkness estimation
적정 조도의 영상을 선별하기위해 다음과 같은 방법을 적용합니다. 먼저 영상에 super-pixel segmentation 을 적용하고, 각 super-pixel의 mean/variance가 일정 threshold 보다 높다면 해당 super-pixel의 밝기는 밝다고 판단합니다. 그리고 밝다고 판단된 super-pixel의 수가 85% 이상이면 해당 영상은 적정 조도 영상으로 판단합니다.

- Blur estimation
뿌옇지 않고 detail이 잘 표현된 영상을 선별하는 과정입니다. 영상에 laplacian filter를 적용하고, 그 결과의 variance 구했을 때, 그 값이 500 이상이라면 detail이 잘 표현된 영상으로 판단합니다.

- Colorfulness estimation
색표현이 잘 된 영상을 선별하는 과정입니다. 고전적인 방법중에 하나[]를 이용하며, 그 값이 500 이상이면 색표현이 잘 된 영상으로 판단합니다.

이 방법을 이용하여 총 344,272 장의 영상중에서  97,030 장 영상을 선택했으며, 이중 1%인 965 장의 영상을 랜덤으로 골라 테스트 셋으로 사용했습니다. 그리고 22,656 장을 포함하여 data-balanced subset을 트레이닝 셋으로 사용했습니다.

### Target image synthesis

선택된 영상들에 대해 실제 저조도 상황과 같은 저조도를 합성하는 과정입니다. 실제 저조도 환경과 유사하게 합성하기 위해

- Low-light image synthesis

$$ I_{out}^{(i)}=\beta \times \left ( \alpha \times I_{in}^{(i)} \right )^{\gamma },i\in \left \{ R,G,B \right \} $$

- Image contrast amplification



## 4. Methodology

### Network architecture

- Attention-Net

- Noise-Net

- Enhancement-Net

- Reinforce-Net

### Loss function

- Attention-Net loss

- Noise-Net loss

- Enhancement-Net loss

- Reinforce-Net loss


## Experimental evaluation

### Experiments on synthetic datasets

### Experiments on real datasets

### Experiments on real Images

### Ablation study

### Unsatisfying cases


## Conclusion


## Reference

