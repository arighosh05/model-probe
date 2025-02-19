<p align="center">
  <img src="./model_probe_logo.jpg" alt="logo" width="100">
</p>


<div align="center">

  <b>Model Probe</b>
----------------------
  a model probe for reasoners

</div>

<p align="center">
  <a href="https://opensource.org/license/unlicense">
    <img src="https://img.shields.io/badge/License-The Unlicense-black.svg" alt="License: The Unlicense">
  </a>
</p>

## About

In an era where algorithms shape policies, headlines, and lives, understanding *how* a machine arrives at its conclusions is more than a technical curiosity—it is a moral imperative. Driven by this belief, **Model Probe** seeks to pull back the curtain on a system’s internal thought processes, highlighting subtle emotional cues, detecting questionable reasoning patterns, and ensuring alignment with ethical standards. The hope is that through such methodical observations, that is, via *probing* the *model*, we might bridge the ontological gap between human and artificial reasoning. This project serves as an existential commitment to maintaining human agency in a world of distributed cognition, and hopes to inspire future work towards more ethical and transparent machine intelligence.


## Architecture

The architecture of the model probe is designed to be modular and adaptable, making it easily generalizable across different reasoning systems.

The core workflow can be summarized by the following high-level diagram:

<p align="center">
  <img src="./high_level_diagram.png" alt="image" width="500">
</p>

The system takes a user query and extracts the model’s internal reasoning content. This reasoning is then fed into three parallel checks: **sentiment analysis**, **rule-based checks**, and **anomaly detection**. The outputs of these checks are aggregated by a **meta-classifier** that assigns an overall concern level. Finally, if certain thresholds are exceeded, the system triggers alerts, ensuring that any potentially problematic or anomalous reasoning is flagged for review.

The rule-based system provides explainable, deterministic detection that complements the more statistical approaches in other parts of the model probe, forming an essential component of the defense-in-depth strategy for monitoring AI reasoning. It serves as the policy enforcement layer, scanning AI reasoning content for **predefined keywords** that may indicate harmful, illegal, or manipulative content. Words are categorized by severity level, allowing for nuanced analysis rather than binary detection.

The sentiment analysis sytem adds an emotional intelligence layer to the model probe that can detect concerning tone and affect that might not be captured by specific keywords, providing a window into the affective characteristics of AI reasoning. The system examines sentiment at both the **document level** and **sentence level**. Special emphasis is placed on identifying negative sentiment, which may indicate concerning reasoning patterns. Sentiment scores are mapped to concern levels based on empirically determined thresholds.

The anomaly detection system serves as the pattern recognition backbone of the model probe, identifying unusual reasoning that might not trigger explicit rules or sentiment flags but still deviates from established norms. The system employs an **Isolation Forest** to identify anomalies in reasoning behavior. While the system may initially appear to be stateless due to its functional pipeline architecture, it actually maintains significant state through two key mechanisms: **Persistent Embeddings Cache** and **Temporal Context**. This enables the system to preserve historical embeddings of reasoning patterns over time, allowing the anomaly detection system to learn and adapt as it processes more instances of reasoning content.

The meta-classifier acts as the decision-making brain of the model probe, synthesizing diverse signals into a unified assessment and actionable classification. It combines outputs from sentiment analysis, rule-based detection, and anomaly detection with appropriate weighting to reflect their relative importance to produce an aggregated result that is gated against carefully calibrated thresholds to map numerical scores to meaningful classification categories. This approach balances **sensitivity** with **specificity**, providing a reliable foundation for the alerting system that follows.

The alerting system forms the action layer of the model probe, incorporating **tiered alert levels**, **complete context inclusion**, and **permanent audit trail** to transform analytical insights into timely notifications for human review when necessary.


## Installation

1. Download [Ollama](https://ollama.com/download).
   
2. Verify that Ollama is working.
   
```
ollama --version
```

3. Download a DeepSeek model of suitable size.

```
ollama pull deepseek-r1:8b
```

4. Initialize the model locally.

```
ollama run deepseek-r1:8b
```

5. Install the following dependencies.

```
pip install ollama nltk sentence-transformers scikit-learn
```

6. Run `script.py`.

```
python script.py "your-prompt-here"
```

## Notes

The following points pertinent to `script.py` bear mentioning. 

- Ensure model consistency.

  Adjust the model that `script.py` via the `model` variable in Line 480.

- Configure email settings.

  Set `EMAIL_ENABLED` to `True` to activate email alerts. Provide `EMAIL_SENDER` and `EMAIL_PASSWORD`. Customize alert recipients in `EMAIL_RECIPIENTS`. Configure `EMAIL_SMTP_SERVER` and `EMAIL_SMTP_PORT` appropriately. Critical alerts will be sent for high-concern classifications and Warning alerts will be sent for medium-concern classifications.

- Customize word list.

  Modify word lists in `HARMFUL_WORDS`, `ILLEGAL_WORDS`, and `MANIPULATION_WORDS` to match your use case. These word lists fuel the rule-based system.

- Verify file structure.

  The script creates `data/ directory` for embedding cache storage and creates `logs/ directory` for detailed analysis logs, saving each analysis as timestamped JSON files.

- Adjust anomaly detection.

  Modify anomaly detection parameters in `initialize_anomaly_detection()`. First-time use will have limited anomaly detection until more samples are collected. Accuracy increases after processing at least 10 samples.

## Meta

Aritra Ghosh – aritraghosh543@gmail.com

Distributed under the The Unlicense license. See `LICENSE` for more information.

[https://github.com/arighosh05/](https://github.com/arighosh05/)

## Contributing

1. Fork it (<https://github.com/arighosh05/model-probe/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
