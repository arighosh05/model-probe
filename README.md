<p align="center">
  <img src="./model_probe_logo.jpg" alt="logo" width="100">
</p>


<div align="center">

  <b>Model Probe</b>
----------------------
  a model probe for reasoners

</div>

<p align="center">
  <a href="https://opensource.org/license/0bsd">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

## About

## Architecture

<p align="center">
  <img src="./high_level_diagram.png" alt="image" width="500">
</p>

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

## Meta

Aritra Ghosh – aritraghosh543@gmail.com

Distributed under the MIT license. See `LICENSE` for more information.

[https://github.com/arighosh05/](https://github.com/arighosh05/)

## Contributing

1. Fork it (<https://github.com/arighosh05/model-probe/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
