# Adaptive kNN Machine Translation Paper Reproduction
## Abstract
In this project, we reproduced a paper about adaptive nearest neighbor machine translation. KNN-MT, recently proposed by Khandelwal et al. (2020a), is a method which combines a pretrained neural machine translation (NMT) model with token-level k-nearest-neighbor (kNN) retrieval to improve the translation accuracy. Our selected paper introduced an adaptive KNN-MT model that determines the choice of k regarding each target token adaptively. In order to reproduce their findings, we reran the models on reported datasets with similar settings mentioned in the paper. We also validated the generalization performance of the model. From our results, we reached the conclusion that in the aspect of performance, the model proposed by this paper does perform better than the original one; but we nonetheless observed incoherence between our generality evaluation outputs and those in the paper.

## Introduction
The model proposed in our selected paper is based on a combined model of a pretrained neural machine translation and a KNN classifier over a datastore of cached context representations and corresponding target tokens. This model provides a simple yet effective strategy to utilize cached contextual information in inference. However, fixing hyperparameter k in all cases raised potential problems: the original model is sensitive to noise and has poor robustness and generalization performance. To handle this problem, our selected paper introduced an adaptive model to dynamically evaluate and utilize the neighbor information conditioned on different targets. To be more specific, instead of using a fixed k, they consider a set of possible Ks that are smaller than an upper bound K. Then with the retrieval results of the current target token, they proposed a light-weight Meta-k Network to estimate the importance of all possible k-Nearest Neighbor results, and obtain the final decision based on which they are aggregated to.

## Scope of reproducibility
There are two main claims that we want to reproduce from the original paper:

1. The adaptive KNN-MT model is able to effectively filter the noises and significantly outperform the vanilla kNN-MT baseline

2. This adaptive method can also improve generality.

## Methodology
### Model description and hyperparameter choosing
We used the fairseq toolkit(Ott et al., 2019) and faiss(Johnson et al., 2017) to replicate the kNN-MT model. WMT’19 German-English news translation task winner model (Ng et al., 2019) was applied as the pre- trained NMT model which is also used by Khandelwaletal(2020a). For kNN-MT, we tune the hyper- parameter λ and report the best scores for each domain. The hidden size of the two-layer FFN in Meta-k Network was set to 32. We directly use the dev set (about 2k sents) to train the Meta-k Network for about 5k steps. We used Adam (Kingma and Ba, 2015) to optimize our model, the learning rate is set to 3e-4, and batch size is set to 32 sentences.

To check the first claim, we reran the code and compared the BLEU score of the 2 main models, the proposed adaptive kNN-MT and the vanilla kNN-MT. In the original paper, they also compared another uniform kNN-MT, where they set equal confidence for each kNN prediction and made it close to the vanilla kNN-MT with small Ks. Due to limited time and computational resources, we decided to not include this model in our experiments. In the original paper, the authors tested the models with max K values 1,2,4,8,16,32. The time for training increased dramatically as k became larger so we only reproduced part of it with k=1,2,4.

For the second claim, we repeated the same steps as mentioned in our selected paper but only with a smaller max K value. We set K=4 in all settings. To check the generality of this adaptive model, we utilized the Meta-k Network trained on each of these 4 domains to evaluate the other three domains. As for robustness, we firstly used one domain(for example medical) set and datastore to tune hyperparameter for vanilla kNN- MT and train the Meta-k Network for adaptive kNN-MT, and then applied the test sets of other domains to test the model with medical datastore.

### Datasets and Evaluation Metric
The original paper included one multi-domain dataset as the pretrained baseline and the other 4 datasets with different domains of IT, Medical, Koran, and Law. We followed the steps in the paper and reran the code on all these datasets to reproduce the results. We used the Moses toolkit to tokenize the sentences and split the words into sub-word units, and ScareBLEU score for model evaluation.

### Computational requirement
For reproducing this paper, we used the virtual machine instance and Colab from Google Cloud Platform. The GPU we used for our virtual machine is NVIDIA Tesla T4 graphics card and CPU with 4 cores. The operating system we used is Cent OS, which supports running shell scripts.

## Results
In the beginning, we tried to run the model with the same settings as in the original paper. However, we found that the default training epoch=500 was too large and took too much time. Therefore, we changed the number of epochs into 50 which obviously increased the training duration while keeping the result scores at the same level as before. During the training process, we saved two models at each checkpoint. One is the model so far with the lowest validation loss, and another one is the model which has trained for the largest epochs. From this perspective, the reduction of the number of epochs will not affect the inference result a lot. For another reason, we have discussed the concept of early stopping in class. The training process should be terminated as early as the model has already converged. It is noticeable that the results we got by using only 50 epochs are similar to the results from the original paper.

The following table1 shows the BLEU scores of the vanilla KNN-MT(V) and adaptive kNN-MT model(A). We can find that the Adaptive kNN-MT model introduced in our selected paper has a better result on almost all domains, which proved the first claim in this paper.

![image](https://user-images.githubusercontent.com/68981504/148321826-e9300828-7601-40ca-a111-7b3fd09b69e8.png)
![image](https://user-images.githubusercontent.com/68981504/148321838-def378b8-8dc4-40d4-a206-a1f5f65f5cae.png)

### Generality
As indicated in the paper, the generality of the Adaptive kNN-MT model is decent. In the original paper, the results of the model trained by the IT domain are applied to other domains, and the results are comparable to those produced by models trained with their respective domains. However, the results of models trained by other domains are not included in the paper. We highly doubt the reason why such results are not shown. From our perspective, there is no reason to explain why they only experiment with the model trained by the IT domain. As a consequence, we tested the generality for models trained by each domain to check if the claim of the author is still held. The main results are shown in the following table:

![image](https://user-images.githubusercontent.com/68981504/148321940-9f46c111-d589-402a-854d-87ccb28343a5.png)

When compared to the corresponding table presented in the paper, it can be observed that although our “in- domain” results remain largely identical to those shown in the original paper, outputs concerning applying the model trained on IT domain to all the other domains displayed considerable discrepancies to the corresponding statistics in the paper. We think this can be attributed to the simple fact that we only tested a subset of k values due to hardware constraints, so while the paper chose k = 32 as the baseline model to test for generality, we can only use k = 4 (since we didn’t test k = 32).

## Conclusion and discussion
In this project, we reproduced the Adaptive kNN-MT model and compared it with the vanilla kNN-MT model to investigate two claims in our selected paper. Through our experiments, we reached some conclusions in the original paper. We found that the new adaptive kNN-MT model can indeed improve the performance of the vanilla kNN-MT model. But for k=4, the generality of the adaptive model was not desirable. We also noticed that decreasing the number of training epochs from 500 to 50 can significantly improve the model efficiency while keeping the performance at a high level.

In the future, we want to rerun the remaining part of this paper to check the results for larger Ks to further validate or challenge the claims of the paper.


The following table1 shows the BLEU scores of the vanilla KNN-MT(V) and adaptive kNN-MT model(A). We can find that the Adaptive kNN-MT model introduced in our selected paper has a better result on almost all domains, which proved the first claim in this paper.

## Citation
Bleu: a method for automatic evaluation of machine translation. Kishore Papineni, Salim Roukos, Todd Ward, and Wei- Jing Zhu. https://github.com/mjpost/sacrebleuNearest Neighbor Machine Translation

Urvashi Khandelwal, Angela Fan, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. https://arxiv.org/abs/2010.00710Search 

Engine Guided Neural Machine Translation Jiatao Gu, Yong Wang, Kyunghyun Cho, Victor O.K. Li https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17282

Sequence to Sequence Learning with Neural Networks Ilya Sutskever, Oriol Vinyals, Quoc V. Le https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html
