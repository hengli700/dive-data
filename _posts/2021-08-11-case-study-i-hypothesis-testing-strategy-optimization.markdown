---
layout: post
title:  "Case study I: hypothesis testing and strategy optimization"
date:   2021-08-11 12:31:20 -0700
categories: experiment-design-recommendation
usemathjax: true
---

This case study is one of the take-home assignment provided by Starbucks for their job candidates. It is a great example of 1) **hypothesis testing** and 2) **machine learning in business analytics**. More specifically, hypothesis testing was performed to evaluate data gathered from A/B testing, and determine levels of significance for the business metrics of interest. Following by the evaluations, the conclusions were further used to inform and optimize Ad campaign strategy to improve the business metrics.

## Background information
The data for this exercise consists of about 120,000 data points split in a 2:1 ratio among training and test files. In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company 0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion. Each data point includes one column indicating whether or not an individual was sent a promotion for the product, and one column indicating whether or not that individual eventually purchased that product. Each individual also has seven additional features associated with them, which are provided abstractly as V1-V7.

## Evaluation metrics
The task is to use the training data to understand what patterns in V1-V7 to indicate that a promotion should be provided to a user. Specifically, the goal is to maximize the following metrics:

* **Incremental Response Rate (IRR)**

IRR depicts how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion. Mathematically, it's the ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group (_treatment_) minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group (_control_).

$$ IRR = \frac{purch_{treat}}{cust_{treat}} - \frac{purch_{ctrl}}{cust_{ctrl}} $$


* **Net Incremental Revenue (NIR)**

NIR depicts how much is made (or lost) by sending out the promotion. Mathematically, this is 10 times the total number of purchasers that received the promotion minus 0.15 times the number of promotions sent out, minus 10 times the number of purchasers who were not given the promotion.

$$ NIR = (10\cdot purch_{treat} - 0.15 \cdot cust_{treat}) - 10 \cdot purch_{ctrl}$$


## Case study workflow
1. Inspect the  data and perform EDA to check for data type, any null values, as well as determine the features and labels for each observations.
2. Analyze the current A/B testing result, and retrieve **IRR** and **NIR** values. This step will include validating group assignment by checking invariant metric (proportion of control group), and checking the significance levels of the obtained evaluation metrics.
3. Draw conclusion of current A/B testing results.
4. Optimize promotion strategy using ML models, and evaluate obtained classifiers.

### Inspect the Data
The training data set contains 84543 observations with 7 features and contains no null values. Each observation corresponds to an individual customer and the features associated with each sample are attributes for that customer. Feature V2 and V3 are continuous quantitative variables. However, V1, V4 through V7 are discrete quantitative variables, indicating those might be categorical variables and need proper encoding in step 4.

```
train_data = pd.read_csv('./training.csv')
train_data.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Promotion</th>
      <th>purchase</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>No</td>
      <td>0</td>
      <td>2</td>
      <td>30.443518</td>
      <td>-1.165083</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>No</td>
      <td>0</td>
      <td>3</td>
      <td>32.159350</td>
      <td>-0.645617</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>No</td>
      <td>0</td>
      <td>2</td>
      <td>30.431659</td>
      <td>0.133583</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>26.588914</td>
      <td>-0.212728</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Yes</td>
      <td>0</td>
      <td>3</td>
      <td>28.044332</td>
      <td>-0.385883</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

```python
train_data.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 84534 entries, 0 to 84533
    Data columns (total 10 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   ID         84534 non-null  int64  
     1   Promotion  84534 non-null  object
     2   purchase   84534 non-null  int64  
     3   V1         84534 non-null  int64  
     4   V2         84534 non-null  float64
     5   V3         84534 non-null  float64
     6   V4         84534 non-null  int64  
     7   V5         84534 non-null  int64  
     8   V6         84534 non-null  int64  
     9   V7         84534 non-null  int64  
    dtypes: float64(2), int64(7), object(1)
    memory usage: 6.4+ MB

### Analyze experiment results
#### Calculate IRR and NIR based on experiment results
This step is fairly straight forward by following the definitions of IRR and NIR. It was determined that:
```
Incremental Response Rate (IRR) from the experiment: 0.00945
Net Incremental Revenue (NIR) from the experiment: -2334.60
```
#### Check invariant metric
In this experiment, the metrics to determine the effect of promotions to customers on the purchasing behavior of this $10 item. Before jumping into analyzing the effect, we first need to check if the assignments of promotion to customers was random, i.e. checking the invariant metric. We can use \alpha = 0.05 as the significance level.

$$
\begin{align}
H_0: p_{ctrl} - 0.5 = 0 \\
H_a: p_{ctrl} - 0.5 \neq 0
\end{align}
$$

There are two approaches we could perform this hypothesis testing via analytical method and simulation. Since the assignment of control and treatment group follow binomial distributions, analytical method can be used. Following the analytical approach, $ p-val = 0.5068 $ is obtained. On the other hand, simulation or non-parametric methods such as bootstrapping does not rely on assumptions on distributions. Based off Law of Large Numbers and its closest cousin Central Limit Theorem, modern computing power enables us to use bootstrapping to obtain a good estimate on test statistics, such as means, differences in means, as well as proportion and difference in proportions.



<img src="{{site.baseurl}}/assets/img/20210810-deeprl/.gif" alt="lunar landing"/>

> Figure 1. Successful landing in simulated OpenAI gym using trained deep reinforcement learning agent.




## Acknowledgments
1. Udacity Data Scientist Nanodegree program
