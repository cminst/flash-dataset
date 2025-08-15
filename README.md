<div align="center">
  <h1> ⚡ FLASH </h1>
  <div>
    <strong>F</strong>ast <strong>L</strong>abelled <strong>A</strong>ction <strong>S</strong>egment <strong>H</strong>ighlights
  </div>
</div>
<br>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/datasets/qingy2024/FLASH-Dataset" style="margin: 2px;">
    <img alt="HuggingFace Dataset" src="https://img.shields.io/badge/HuggingFace-000000?style=for-the-badge&logo=huggingface&logoColor=000&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Overview
FLASH is an open-source video evaluation dataset that tests a model's ability to identify the peak frames in a video given a user's prompt, for example, "The moment when the person lands on the ground". This is a much more challenging task than standard temporal action localization.

## Analysis

We compare our dataset to the THUMOS'14 temporal action localization evaluation dataset. Our results are as follows:

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66d78facde54fea8a009927e/Xrkq9zn_k90XhMR8OH1SD.png" width="900px">
</div>

```
                      +----------------------------------------------------+
                      |          Statistics for Peak Duration              |
+-----------+---------|--------+----------+-------+-------+-------+--------|
| dataset   |   count |   mean |   median |   q25 |   q75 |   min |    max |
|-----------+---------+--------+----------+-------+-------+-------+--------|
| FLASH     |    1526 |   0.18 |     0.10 |  0.03 |  0.19 |  0.00 |   4.67 |
| THUMOS'14 |    3007 |   4.04 |     2.90 |  1.50 |  5.50 |  0.20 | 118.10 |
+-----------+---------+--------+----------+-------+-------+-------+--------+
```

## Acknowledgements
This dataset is heavily built on the [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset.
