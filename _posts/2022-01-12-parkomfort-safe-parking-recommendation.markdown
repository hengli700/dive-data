---
layout: post
title:  "ParKomfort: Safe Parking Recommendation"
date:   2022-01-12 12:34:56 -0700
categories: data-science
usemathjax: true
---

[comment]: <> (<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/customer-churn.png" alt="customer churn"/>)

[comment]: <> (> Image credit — www.humanlytics.co)

## 1 INTRODUCTION AND MOTIVATION

Finding a street parking spot during peak hours is always headache due to availability, price and safety concerns. The ideal parking spot would provide sense of
comfort and safety.

Current parking spot-finding products (Parkopedia
and ParkMobile) offer only location and price data with
no regard for crime or policing. Users have to rely on
other systems to search for safer parking options, thus
creating a non-unified experience. This project considers the street parking tickets in a different perspective,
as well as the crime elements, and tries to bring peace
of mind to the users.

The objective of this project is to develop a parking
recommendation application based on user preferences
to make data-informed decisions to locate a parking
spot in big cities such as NYC. The application will
utilize multiple publicly available datasets (NYC parking
violation, parking meter dataset, and NYPD complaint
data) to recommend parking spots based on policing
and crime risk factors.

## 2 LITERATURE SURVEY
### 2.1 Parking recommendation algorithms and designs
Most of the literature uses the spatiotemporal analysis approach to predict availability of on-street parking
either through real-time data collected by infrastructures such as parking meters and parking lots, or build-
ing machine learning models trained on large datasets.
This first approach heavily relies on live data to retrieve
available parking space near the user’s location. It offers
personal preferences, such as walking distance, driving
distance, available parking space, and parking fees. Furthermore, users can adjust the weight to get customized
experience.[Chen and Chiu 2017] Many recommendation systems have been developed. In our project, We
improve on current parking recommendation services
by including risk factors (crime rate and policing) into
consideration.

When real time data is not readily available, machine
learning models trained on large data sets are often used
to make predictions of parking space availability and
recommend to the users.[Awan et al. 2020] In particular,
spatial analysis units (i.e., point, street, census tract, and
grid) are used to examine the impact of spatial scale in
classic machine learning predictive models.[Gao et al.
2019]

### 2.2 Definition of risk factor and how it is measured
In this project , the risk factor is related to the amount of
crime and policing. Crime-related risk is directly associ-
ated with personal/property loss due to auto burglary,
vehicle theft, and vandalism. Policing-related risk comes
into play when the driver forgets the parking restriction
or is not available to move the vehicle, which leads to
tickets or towing charges.

There are many time and location-related risk fac-
tors, such as the type of parking: street, underground
parking lot, as well as the visibility and accessibility of
the parking location, time and duration of the parking,
since night reduces visibility, and duration increases the
number of chances presented to criminals to execute a
crime or for police to issue a fine[Nourinejad et al. 2020].
Subsequently, higher valued vehicles present better targets for criminals (and police). Finally, socio-economic
factors such as population density, demographics, local
culture, economic conditions, as well as attitude toward
crime all affect crime rate[FBI 2011].

Spatial analysis has been used to identify the hot or
cold spots. Kawamura et al. used Getis-Ord G statistics
to identify hot spots of high truck parking violations in
the Chicago metro area, and showed density variation
across downtown to suburbs. G statistic for spatial visualization could be potentially incorporated into risk
factor calculation in the parking recommendation and
visualize the amount of crime and policing [Kawamura
et al. 2014].

## 3 PROPOSED METHOD
### 3.1 Intuition
Since no current methods provide security outlook for
parking, our method provides inherently better options
for parking selection. Our project will help minimize
the chances of negative events (such as vehicle related
crime or parking ticket) and provide sense of safety to
the users.

### 3.2 List of innovation
- Providing security outlook for parking
- Providing customizable User Experience to accommodate individual risk tolerance of all users

### 3.3 Approach
#### 3.3.1 Data gathering
The data used for this project came from 3 different
source, i.e. NYC parking tickets (dataset 1), NYC street
parking meter (dataset 2), and NYPD complaint data
(dataset 3).

Dataset 1 is the New York parking tickets data from
NYC OpenData for the fiscal year 2021 and 2022 (about
1.2 GB), which contains tickets issued in year 2020 and 2021. It provides locations of parking violations as well
as valid parking duration. Combined with dataset 2,
it provides available parking spaces near destinations.
Using dataset 3 we extracted vehicle-related crimes near
the destination. With these datasets, information related
to parking availability and risk factors can be derived
and used in parking recommendation.

#### 3.3.2 Data wrangling
The crime database contained all complaint data of
NYPD, which needed to be filtered for vehicle related
entries only. Additionally, even though dataset 1 and
dataset 2 has large amount of data, some are unusable.
The data is deemed unusable for several reasons: 1) The
locations are scattered all over the world, even if the
data is filtered to display NY license place. 2) incomplete
addresses (missing city, or state, or house number). 3)
ticket issuance date in the future or several years back.
The approach to filtering out those data are 1) removing the address not in NYC, 2) try to get the geolocations if
we have enough information such as street/city/state, regardless of the house number, and 3) Focus on the ticket
issuance date of the given fiscal year. After filtering
out unusable data, we focused on getting geolocations
and street addresses for the project. Unfortunately, the
dataset 1 (parking tickets database) provides only street
addresses. We used python library geopy to convert the
street addresses to latitude/longitude pairs. However,
the throughput of the API severely limited the amount
of the data that can be processed within the project timeline.

A few methods were used to try to get as much
conversion as possible: 1. Getting the list frequently appeared street name and then run them in the order of
most to least. 2. Run the different set of street names
on different machines simultaneously. 3. Make sure that
each request have at least one second apart in between
so it does not overwhelm the API. They worked. Similar issue happened with dataset 2(NYC parking meter
database) as it only has geolocations and no address. We
then run a reverse search using the same python geopy
library.

#### 3.3.3 Data Analysis.
We also implemented an unsupervised machine learning algorithm, k-means clustering, to identify similarities among the parking ticketing data. Each parking
citation will be assigned to a cluster and will map to
police precincts. It will enable the calculation of the distribution of tickets per precinct within a cluster, allow-
ing us to identify the precincts which are more likely to
issue parking violations.[Lin et al. 2019]. The heat map
visualization of parking violations will also be produced
for in-depth analysis and data-driven decision-making.

The cluster technique of the k-means algorithm can
also be applied to the crime data to identify the crime
trend and zoning knowledge[Thota et al. 2017][Wang
et al. 2020].The k-means algorithm will be implemented
as shown in Algorithm 1. To determine the optimum
number of clusters, we graph the relationship between
the number of clusters and scaled within cluster sum
of squared errors (WCSS) then we select the number of
clusters where the change in WCSS begins to level off
(elbow method).[Lin et al. 2019].

$$ WCSS_{k} = sum_{i=1}^{k} \frac{1}{2n_{i}}D_{i}$$



\begin{algorithm}
\caption{$k$-Means Algorithm}\label{alg:cap}
\begin{algorithmic}
\State Initialize cluster centroids $u_1$, $u_2$, ..., $u_k$  $\epsilon$ $R^m$ randomly
\While{$u_j$ $is$ $not$ $converged$}
\ForEach {$i$}
\State $u^{(i)} := arg min_j || x^{(i)} - \mu_j||^2$
\EndFor
\ForEach {$j$}
\State $\mu_j := \frac{\Sigma_{i=1}^m \{c^{(i)} = j\} x^{(i)} }{\Sigma_{i=1}^m \{c^{(i)} = j\}}$
\EndFor
\EndWhile

\end{algorithmic}
\end{algorithm}

\begin{equation}
WCSS_k = \sum\limits_{k=1}^k  \frac{1}{2n_k} D_k
\label{eqn:kmeans_assign_step}
\end{equation}
where
\begin{equation}
D_k = {\sum\limits_{x_i \epsilon C_k} \sum\limits_{x_j \epsilon C_k}  ||x_i - x_j||^2 } = {2n_k \sum\limits_{x_i \epsilon C_k} ||x_i - \mu_k||^2}
\label{eqn:kmeans_assign_step2}
\end{equation}
\newline


## What is this project about?

Sparkify is a fictional music streaming service similar to Apple Music, Spotify, etc. The dataset provided is in the form of event logs when customers use the streaming services, including artists, songs, geolocation, timestamp, demographic infos, user actions etc.

The main goal of this project is to analyze the event log data and build a machine learning model to predict the users who are more likely to cancel service (churn), and thus use this information in ad campaign to reduce customer out-flow.

In addition, due to the amount of data used, we leveraged spark (pySpark), spark SQL, spark ML and cloud service provided by IBM to perform data wrangling, exploration, and machine learning tasks.

## Data exploration

The entire dataset contains 543705 rows, with following columns corresponding to the record from the event log. Among these rows, there are 15700 rows with empty userId from “Logged Out” and “Guest” users. The data set was first cleaned by removing those rows and only considered rows with valid userId.

```
root
 |-- artist: string (nullable = true)
 |-- auth: string (nullable = true)
 |-- firstName: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- itemInSession: long (nullable = true)
 |-- lastName: string (nullable = true)
 |-- length: double (nullable = true)
 |-- level: string (nullable = true)
 |-- location: string (nullable = true)
 |-- method: string (nullable = true)
 |-- page: string (nullable = true)
 |-- registration: long (nullable = true)
 |-- sessionId: long (nullable = true)
 |-- song: string (nullable = true)
 |-- status: long (nullable = true)
 |-- ts: long (nullable = true)
 |-- userAgent: string (nullable = true)
 |-- userId: string (nullable = true)
```

Before further exploring the data, it is necessary to define what “churn” means for each user. Related back to the indication of churn as cancelling a service, after obtaining distinct actions in the “page” column (shown below), we define the occurrence of churn event when a user has action of “Cancellation Confirmation”. This definition applies to both paid and free tier customers.

```
['Cancel', 'Submit Downgrade', 'Thumbs Down', 'Home', 'Downgrade', 'Roll Advert', 'Logout', 'Save Settings', 'Cancellation Confirmation', 'About', 'Settings', 'Add to Playlist', 'Add Friend',
'NextSong', 'Thumbs Up', 'Help', 'Upgrade', 'Error', 'Submit Upgrade']
```

For users with churn event, users were marked with 1 in a separate column named “churn”. With the definition in place, we can perform more analyses on how Sparkify customers interact with the service based on whether they are churned or remaining customers.

### General trend


<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/user-trend.png" alt="user trend"/>

> Figure 1. (top) Trend of distinct active users over month. (bottom) Number of actions for selected users in Oct and Nov 2018.

The presented dataset spans from the beginning of Oct 2018 to beginning of Dec 2018. It is helpful to first get an overview of how Sparkify business in terms of month active users.

Figure 1 (top) shows there is a slight drop in monthly active users from Oct to Nov 2018. With the amount of data on hand, it is inconclusive whether this change is significant or not. However, it does signaling the need to look at customer churn. On an individual level, Figure 1 (bottom) shows some users increased service usage from Oct to Nov; others decrease or even not use the service at all.

### Churned users break-down by categorical features

The proportion of churned users (churned proportion) among all distinct users is around 0.2210 or 22.10%.

Breaking down churned customers and remaining customers via demographic informations such as gender, it was found that *female users have slightly higher churn proportion than male users (p_female=0.2273 vs p_male =0.2160)*. However, after running two-sided proportions z-test, **the difference in churned proportion between female and male users is not significant using a significance level of 0.05**.

Similarly, we can break down churned vs remaining customer group by paid or free tiers. It was found *free-tier users have slightly lower churn proportion than paid-tier users (p_free=0.2216 vs p_paid=0.2336)*. After running hypothesis testing, **the difference in churn proportion between free-tier and paid-tier users is not significant using a significance level of 0.05**.

### Churned users break-down by numerical features

The numerical features used in this project were selected from a subset of columns (“ts”, “registration_time” and “page”). Combined with timestamp of churn event, “registration_time” was used to calculate the tenure or customer lifetime in days. “page” column was used to generate user actions with the service, and the total number of each action for each user at each level was aggregated. The following numerical features were used, and selected rows with numerical features is shown in Figure 2. **Note**: *there are multiple users who have been both paid and free users due to upgrade and downgrade events*.

```
'tenure_days', 'total_session', 'total_thumbs_down', 'total_home', 'total_roll_advert', 'total_logout', 'total_save_settings', 'total_about', 'total_settings', 'total_add_to_playlist', 'total_add_friend', 'total_nextsong', 'total_thumbs_up', 'total_help', 'total_error'
```

<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/numerical-features.png" alt="numerical features"/>

> Figure 2. Numerical features for selected row.

First, let’s take a look at the some statistics of these numerical features for churned and remaining users.

<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/boxplot-hist.png" alt="boxplots and histplots"/>

> Figure 3. (left) Boxplots showing variabilities and (right) histplots showing distributions of numerical features for remaining (0) and churned (1) users.

Figure 3 shows the variabilities and distributions of these numerical features. The median values for remaining and churned users are similar in all numerical features, and they all have a some amount of outliers signaling skewed distributions.

The skewness is also indicated by the distribution plots. The direction of skewness is also worth studying. For some features, such as tenure days, total_add_to_playlist, total_add_friend, total_nextsong, it would be preferred to have left-skewed distributions with customers staying longer and have more positive engagements with the services. Apparently, this is not the case for the distributions from presented dataset. On other hand, for features like total_error and total_thumbs_down, a right-skewed distribution is preferred signaling fewer negative experiences.

In general, remaining users interacts more with the service across all events compared with churned users. Due to nature of aggregated properties and potential correlation between page actions, it is needed to explore the correlations among different features.

<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/cor1.png" alt="correlation matrix 1"/>

> Figure 4. Correlation matrix among different numerical features.

Figure 4 shows the correlation matrix among different numerical features. We could see that strong correlations exists! Proper feature engineering is needed to reduce correlations, in order to reduce multicollinearity and potential issues when using regression models.

## Feature engineering

Feature engineering was first performed by normalizing aggregated features by total_sessions. After normalization, obtained correlation matrix is shown in Figure 5.

<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/cor2.png" alt="correlation matrix 2"/>

> Figure 5. correlation matrix after normalization.

Some features are still highly correlated, such as add_to_playlist_per_session, home_per_session, thums_up_per_session, and next_song_per_session. These are all related to events implying positive user experience. We could add those feature together into a new feature, named postive_engagement_per_session. Updated correlation matrix is shown in Figure 6.

<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/cor3.png" alt="correlation matrix 3"/>

> Figure 6. Updated correlation matrix after combining several normalized features.

The numerical features after feature engineering showed less correlation. The transformed data set is then fed into machine learning pipeline to train models to make predictions on customer churn.

## Modeling

The transformed dataset was first split into train, test, and validation sets. Several of the machine learning classifiers, including Logistic Regression, Decision Tree, Random Forest, and Gradient Boost Tree, were used for training. Since the proportion of churned users is small compared with remaining users, i.e. imbalanced dataset, F1 score was used as the main measure for accuracy of various models.

<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/base.png" alt="baseline performance"/>

> Figure7. Baseline performances of various ML models on validation set.

Before feeding the training set into ML models, feature scaling through standardization was performed. Baseline performances were first obtained for the above models using default settings from Spark ML library. The result is shown in Figure 7.

From the baseline score result, random forest classifier performed the best on validation set with F-1 score of 0.7761 and accuracy of 0.815.

Use the same trained random forest classifier on test set, we got F-1 score of 0.7007 and accuracy of 0.76. To improve its performance, hyper-parameter tuning was performed on this model with 5-fold cross-validation on the training set.

After, hyper-parameter tuning, the performance of the newly trained random forest classifier on test set improved slightly to a F-1 score of 0.7066 and accuracy of 0.768.


<img src="{{site.baseurl}}/assets/img/20210825-spark-churn/importance.png" alt="feature importance ranking"/>

> Figure 8. Feature importance ranking extracted for tuned random forest classifier.

Feature importance was also extracted from tuned model (shown in Figure 8). Looking at feature importance ranking, we could see that customer tenure time is the most important, followed by user actions, such as receiving ads, thumb-downs, and positive engagements (add to playlist, nextsong, thumbs up). On the other side of the spectrum, gender and level of service are of least importance, which is in line with the hypothesis testing where we conclude the differences are not significant.

## Conclusion

In this project, we leveraged Spark to analyze large amount of user logs and built machine learning models to predict customer churn, with the intention of using trained model to be potentially serving future ad campaigns.

By performing hyper-parameter tuning, we achieved slight improvement in F-1 scores on the test set using the tuned random forest classifier. The limited performance improvement might be due to the following reasons:

- Imbalanced dataset (99 churned vs 349 remaining unique users). Even though we use F-1 score as metric to reduce the bias from imbalanced data, rebalancing could be performed to further improve model performance.

- Information loss during feature engineering stage. To reduce the multicollinearity, we normalized and combined some features. Even though it could reduce the issues for regression models, it might be worth investigating using features as is for models more resilient to multicollinearity, such as random forest, decision tree.

- Need for more features extracted from event log. There are more information from the event log that has not been used in this project, such as geolocation tags and downgrade/upgrade related events. These new features could offer more dimensions to correctly predict customer churn.

To see more about this analysis, please direct to my Github repository <a href="https://github.com/hengli700/spark-machine-learning-sparkify" target="_blank">here</a>.

---

## Acknowledgments
- Datasets are provided by Udacity Data Scientist Nanodegree program
- Cloud service provided by IBM Cloud
