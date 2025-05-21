# BioMAISx

This repository releases the BioMAISx (*Bio*technology: *M*edia, *A*griculture, *I*nvestment, (and) *S*entiment E*x*cerpts) dataset annotated for Aspect-Based Sentiment Analysis (ABSA). It includes all code required for collecting and processing the raw data used for annotation, details on how the data was annotated, and code for post-processing the annotated data. 

The dataset is made available as a csv [here](https://drive.google.com/file/d/1PmJr1DholnD-Bnr_Q4FlcRajRHZ5kNJs/view?usp=sharing). See [here](https://github.com/uchicago-dsi/BioMAISx/blob/main/annotation/BioMAISx_polarityDistribution.pdf) the polarity distribution per aspect category.

A Zenodo link will later be made available. 

Examples of preparing and using this data to train ABSA models is located in tutorials. 

## Collecting Data

The raw articles from which the quotes used in this corpus were sourced came from Factiva. You need to gain access to articles from Factiva (for a fee) and attain a user key and CID. Then to download the articles, set your key and CID to environment variables named `FACTIVA_USER_KEY` and `FACTIVA_CID`, respectively. Then you should be able to successfully run `python scripts/download-source.py`

## Preprocessing Data

From the raw text data, we filtered to articles with specific keyterms, extracted quotations from those articles, and then filtered those quotations to those within contianing terms from the desired lexicon. 

From this the quotes were reformatted for annotation with LabelStudio and proposed entities (noun chunks) were extracted using SpaCy. The code for this transformation is in scripts/preprocess-source.py

## Annotating

Relevant information and code for annotation is included in annotation/README.md

If you find this repository helpful, feel free to cite our publication [BioMAISx: A Corpus for Aspect-Based Sentiment Analysis of Media Representations of Agricultural Biotechnologies in Africa:](https://dl.acm.org/doi/abs/10.1145/3627673.3679152)

```
@inproceedings{chiril2024biomaisx,
  title={BioMAISx: A Corpus for Aspect-Based Sentiment Analysis of Media Representations of Agricultural Biotechnologies in Africa},
  author={Chiril, Patricia and Spreadbury, Trevor and Rock, Joeva Sean and Dowd-Uribe, Brian and Uminsky, David},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={5338--5342},
  year={2024}
}
```
and [Seeds of Discourse: A Multilingual Corpus of Direct Quotations from African Media on Agricultural Biotechnologies:](https://aclanthology.org/2025.findings-naacl.473.pdf)

```
@inproceedings{chiril2025seeds,
  title={Seeds of Discourse: A Multilingual Corpus of Direct Quotations from African Media on Agricultural Biotechnologies},
  author={Chiril, Patricia and Spreadbury, Trevor and Rock, Joeva Sean and Dowd-Uribe, Brian and Uminsky, David},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={8494--8500},
  year={2025}
}
```
