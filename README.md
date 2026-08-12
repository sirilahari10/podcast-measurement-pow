# podcast-measurement-pow
# Video-Podcast Incrementality & GenAI Translation Framework

## Overview
This repository contains a Proof of Work (PoW) designed for the Spotify Podcast & Video Analytics team. It demonstrates a lightweight causal inference framework to measure the impact of video podcasts, paired with a Generative AI layer to translate statistical results into actionable insights for cross-functional stakeholders (editorial, marketing, and studios).

## The Business Problem
When a creator adds a video component to a highly successful audio podcast, does it drive **incremental total consumption hours**, or does it simply cannibalize the existing audio-only listening time? 

To measure this effectiveness without bias, we cannot simply compare video listeners to audio listeners. We need a causal framework.

## Methodology: Difference-in-Differences (DiD)
This project uses a Difference-in-Differences approach on synthetic user-level streaming data. We observe a control group (users who only received the audio feed) and a treatment group (users who received the new video feed) across two time periods (Pre-Launch and Post-Launch).

The regression model is defined as:
$$Y_{it} = \beta_0 + \beta_1 \text{Treatment}_i + \beta_2 \text{Post}_t + \beta_3 (\text{Treatment}_i \times \text{Post}_t) + \epsilon_{it}$$

Where $\beta_3$ represents the true causal estimate of the video feature on total streaming minutes.

## GenAI Dashboard Integration
A common bottleneck in analytics is translating causal inference output (p-values, confidence intervals, coefficients) into language that editorial and content strategy teams can confidently act on. 

This project includes a Python module that ingests the `statsmodels` OLS summary and uses an LLM to generate a plain-English "Executive Summary" ready for an automated dashboard.
