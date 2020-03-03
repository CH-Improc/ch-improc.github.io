---
title: Kaggle competition - Bengali AI
data: 2020-03-04
description:
MNIST의 벵골어 버전 competition, 약 13000개의 클래스
categories: 
- Kaggle competition

tags:
- Image classification
- handwritten recoginition
---

# Introduction

MNIST의 벵골어 버전  
[Bengali.AI competition 페이지](https://www.kaggle.com/c/bengaliai-cv19/overview)


- 벵골어는 세계에서 5번째로 많이 사용하며, 방글라데시와 인도에서 많이 사용함

- 49개의 문자(자음 38개, 모음 11개)로 이루어져 있는데 18개의 분음 부호와 악센트로 이루어져 있어 복잡하며, OCR 하기 어려운 문제가 있음

- 비영리 기관 https://bengali.ai/ 에서 대량의 dataset을 구축했으며, 벵골어 인식 연구에 가속화를 위해 공개함

- 벵골어의 문자 하나는 grapheme root, vowel diacritics, consonant diacritics 세개의 요소로 나눌 수 있으며 grapheme root 요소는 168개의 클래스, vowel diacritics 요소는 11개의 클래스, consonant diacritics 요소는 7개의 클래스로 분류됨(아래 그림 확인)

<figure><img src="{{ '/assets/post_images/Bengali_AI_figures/fig01.png' | prepend: site.baseurl}}" width="67%" height="67%" alt="fig01"></figure>

# Reference

<https://www.kaggle.com/c/bengaliai-cv19/overview>