# Deep Learning Visual Attack

## Introduction

This is a project to implement white-box and black-box attacks on deep learning visual models, using python and some popular libraries and frameworks, such as numpy, pandas, matplotlib, tensorflow, pytorch, gym, etc.

## Purpose

The purpose of this project is to learn and practice how to generate adversarial examples, which are images or videos that have been slightly modified to fool deep learning models, but are imperceptible to human eyes. The project also aims to evaluate the effectiveness of different attack methods, such as FGSM, MI-FGSM, NI-FGSM, etc., and to explore some novel approaches to improve the transferability of adversarial examples, such as ensemble training, data augmentation, meta-learning, etc.

## Method

The project uses reinforcement learning (DRL) as the deep learning visual model, and chooses three game environments (Pong, Cartpole, Breakout) from gym as the visual tasks. The project trains DRL agents on these environments using different DRL algorithms (Deep Q-Learning, Policy Gradients), and then uses white-box attack methods (FGSM, MI-FGSM, NI-FGSM) to generate adversarial examples, aiming to test the transferability of adversarial examples across different algorithms and policies. The project also tries some novel approaches to boost the transferability of adversarial examples, such as using better optimization, ensemble training, data augmentation, meta-learning, etc.

## Result

The project shows that adversarial examples can successfully fool DRL agents and degrade their performance, and that different attack methods have different success rates, transferability, and perturbation sizes. The project also shows that some novel approaches can improve the transferability of adversarial examples, but also have some limitations and challenges.

## Analysis

The project analyzes the advantages and disadvantages of different attack methods, as well as the possible reasons and impacts of the attack results. The project also discusses some interesting practical scenarios to apply the attacks, such as autonomous driving, CAVs, etc., and some possible ways to mitigate the effects of adversarial attacks, such as adversarial training, detection, etc.

## Conclusion

The project concludes that deep learning visual models are vulnerable to adversarial attacks, and that generating and transferring adversarial examples is a challenging and fascinating topic. The project also suggests some future directions and open questions for further research and exploration.
