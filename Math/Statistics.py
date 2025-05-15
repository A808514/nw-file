# 1- Statistics For Data Science

# ((((((((((((((( Descriptive Statistics )))))))))))))))

# ( Mean )

# import numpy as np # you can get ( Mean ) by this library
# import statistics # you can get ( Mean ) by this library, this best from the first library

# arr = [10, 15, 20, 25, 30]

# mean = np.mean(arr)
# print(f"Mean: {mean}")

# mean = statistics.mean(arr)
# print(f"Mean: {mean}")

# ------------------------------------------------------------------------------

# ( Mode )

# import scipy.stats as stats # you can get ( Mode ) by this library
# import statistics # you can get ( Mode ) by this library, this best from the first library

# arr = [2, 4, 1, 9, 6, 4, 3, 4]

# mode = stats.mode(arr)
# print(f"Mode: {mode}")

# mode = statistics.mode(arr)
# print(f"Mode: {mode}")

# ------------------------------------------------------------------------------

# ( Median )

# import numpy as np # you can get ( Median ) by this library
# import statistics # you can get ( Median ) by this library, this best from the first library

# arr = [2, 4, 1, 9, 6, 5, 3]

# median = np.median(arr)
# print(f"Median: {median}")

# median = statistics.median(arr)
# print(f"Median: {median}")

# ------------------------------------------------------------------------------

# ((((((((((((((( Measure of Variability : Understanding Data Dispersion )))))))))))))))

# ( Range )
# Range = Largest data value – smallest data value

# arr = [10, 40, 50, 90, 120, 60]

# maximum = max(arr)
# minimum = min(arr)

# Range = maximum - minimum

# print(f"Maximum = {maximum} and Minimum = {minimum} and Range = {Range}")

# ------------------------------------------------------------------------------

# ( Variance )
# average squared deviation from the mean

# import statistics

# arr = [10, 40, 50, 90, 120, 60]
# variance = statistics.variance(arr)

# print(f"Variance = {variance}")

# ------------------------------------------------------------------------------

# ( Standard Deviation )
# measure the extent of variation or dispersion in data

# import numpy as np
# import statistics

# arr = [10, 40, 50, 90, 120, 60]
# std1= statistics.stdev(arr)
# std2 = np.std(arr)

# print(f"Standard deviation 1: {std1}")
# print(f"Standard deviation 2: {std2}")

# ------------------------------------------------------------------------------

# ((((((((((((((( Measure of Shape )))))))))))))))

# ( Skewness )
# the measure of asymmetry of probability distribution about its mean.

# import numpy as np
# import statistics
# def skewness(data):
#     mean_value = np.mean(data)
#     std_dev = np.std(data)
#     n = len(data)
#     skew = (sum((x - mean_value) ** 3 for x in data) * n) / ((n - 1) * (n - 2) * std_dev ** 3)
#     return skew

# data = np.array([2.5, 3.7, 6.6, 9.1, 9.5, 10.7, 11.9, 21.5, 22.6, 25.2])

# print(f"Skewness: {skewness(data)}")

# ------------------------------------------------------------------------------

# ((((((((((((((( Types of Probability Functions )))))))))))))))

# ( CDF )

# import numpy as np
# import matplotlib.pyplot as plt

# n = 500

# data = np.random.randn(n)

# count, bins_count = np.histogram(data, bins = 10)

# pdf = count / sum(count)

# cdf = np.cumsum(pdf)

# plt.plot(bins_count[1:], pdf, color = "red", label = "PDF")
# plt.plot(bins_count[1:], cdf, label = "CDF")
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------

# ( CDF )
# Using data sort

# import numpy as np
# import matplotlib.pyplot as plt

# n = 500

# data = np.random.randn(n)

# x = np.sort(data)

# y = np.arange(n) / float(n)

# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.plot(x, y, marker = "o")
# plt.show()

# ------------------------------------------------------------------------------

# ((((((((((((((( Probability Distributions Functions )))))))))))))))

# Normal or Gaussian Distribution
# ( How to calculate probability in a normal distribution given mean and standard deviation in Python? )

# from scipy.stats import norm
# import numpy as np

# data_start = 5
# data_end = -5
# data_points = 11

# data = np.linspace(data_start, data_end, data_points)

# mean = np.mean(data)
# std = np.std(data)

# probability_pdf = norm.pdf(3, loc = mean, scale = std)

# print(probability_pdf)

# ------------------------------------------------------------------------------

#  ( Python Implementation of t-Distribution )

# # Creating Random Values Using Student’s T-distribution

# from scipy.stats import t
# import numpy as np
# import matplotlib.pyplot as plt

# a, b = 4, 3

# rv = t(a, b)
# random_values = rv.rvs(size=5)

# print(f"Random values: {random_values}")

# # Student’s T-Distribution  Continuous Variates and Probability Distribution

# quantile = np.arange(0.01, 1, 0.1)

# r = t.rvs(a, b)  # random variates
# print(f"Random variates: {r}")

# pdf = t.pdf(a, b, quantile)
# print(f"Probability distribution: {pdf}")

# # Graphical Representation of Random Values Created Using T-Distribution.

# distribution = np.linspace(0, np.minimum(rv.dist.b, 6))
# print(f"Distribution: {distribution}")

# plt.plot(distribution, rv.pdf(distribution))
# plt.show()

# # T-Distribution Graph With Varying Positional Arguments

# x = np.linspace(0, 5, 100)

# y1 = t.pdf(x, 1, 3)
# y2 = t.pdf(x, 1, 4)

# plt.plot(x, y1, "*", x, y2, "r--")
# plt.show()

# ------------------------------------------------------------------------------

# ( T-Distribution Graph With Varying Degrees of Freedom )

# from scipy.stats import t
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(-5, 5, 100)
# degrees_of_freedom = [1, 2, 5, 10]

# for df in degrees_of_freedom:
#     y = t.pdf(x, df)
#     plt.plot(x, y, label=f"Degrees of freedom: {df}")

# plt.xlabel("X")
# plt.ylabel("PDF")
# plt.title("T-Distribution with Varying Degrees of Freedom")
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------

# ( Python – Non-Central Chi-squared Distribution in Statistics )

# # Creating non-central chi-squared continuous random variable

# from scipy.stats import ncx2
# import numpy as np
# import matplotlib.pylab as plt

# a, b = 4.32, 3.18
# rv = ncx2(a, b)
# print(f"RV: {rv}")

# # non-central chi-squared continuous variates and probability distribution

# quantile = np.arange(0.01, 1, 0.1)

# r = ncx2.rvs(a, b)
# print(f"Random variates: {r}")

# r = ncx2.pdf(a, b, quantile)
# print(f"Probability distribution: {r}")

# # Graphical Representation

# distribution = np.linspace(0, np.minimum(rv.dist.b, 20))
# print(f"Distribution: {distribution}")

# plt.plot(distribution, rv.pdf(distribution))
# plt.show()

# # Varying Positional Arguments

# x = np.linspace(0, 5, 100)

# y1 = ncx2.pdf(x, 1, 3)
# y2 = ncx2.pdf(x, 1, 4)

# plt.plot(x, y1, "*", x, y2, "r--")
# plt.show()

# ------------------------------------------------------------------------------

# ((((((((((((((( Parameter estimation for Statistical Inference )))))))))))))))

# ( ML | Expectation-Maximization Algorithm )
# an iterative method used in unsupervised machine learning to estimate unknown parameters in statistical models. It helps find the best values for unknown parameters, especially when some data is missing or hidden.
# This process repeats until the model reaches a stable solution, improving accuracy with each iteration. EM is widely used in clustering

# import numpy as np
# from scipy.stats import norm
# from scipy.stats import gaussian_kde
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Generate a dataset with two Gaussian components
# mu1, sigma1 = 2, 1
# mu2, sigma2 = -1, 0.8

# x1 = np.random.normal(mu1, sigma1, size=200)
# x2 = np.random.normal(mu2, sigma2, size=600)
# x = np.concatenate([x1, x2])

# sns.kdeplot(x)
# plt.xlabel("X")
# plt.ylabel("Density")
# plt.title("Density Estimation of X")
# plt.show()

# # Initialize parameters
# mu1_hat, sigma1_hat = np.mean(x1), np.std(x1)
# mu2_hat, sigma2_hat = np.mean(x2), np.std(x2)
# pi1_hat, pi2_hat = len(x1) / len(x), len(x2) / len(x)

# # Perform EM algorithm
# num_epochs = 20
# log_likelihoods = []

# for epoch in range(num_epochs):
#     gamma1 = pi1_hat * norm.pdf(x, mu1_hat, sigma1_hat)
#     gamma2 = pi2_hat * norm.pdf(x, mu2_hat, sigma2_hat)
#     total = gamma1 + gamma2
#     gamma1 /= total
#     gamma2 /= total

#     mu1_hat = np.sum(gamma1 * x) / np.sum(gamma1)
#     mu2_hat = np.sum(gamma2 * x) / np.sum(gamma2)
#     sigma1_hat = np.sqrt(np.sum(gamma1 * (x - mu1_hat) ** 2) / np.sum(gamma1))
#     sigma2_hat = np.sqrt(np.sum(gamma2 * (x - mu2_hat) ** 2) / np.sum(gamma2))
#     pi1_hat = np.mean(gamma1)
#     pi2_hat = np.mean(gamma2)

#     log_likelihood = np.sum(
#         np.log(
#             pi1_hat * norm.pdf(x, mu1_hat, sigma1_hat)
#             + pi2_hat * norm.pdf(x, mu2_hat, sigma2_hat)
#         )
#     )
#     log_likelihoods.append(log_likelihood)

# plt.plot(range(1, num_epochs + 1), log_likelihoods)
# plt.xlabel("Epoch")
# plt.ylabel("Log likelihood")
# plt.title("Log likelihood vs Epoch")
# plt.show()

# # Plot the final estimated density
# x_sorted = np.sort(x)
# density_estimation = pi1_hat * norm.pdf(
#     x_sorted, mu1_hat, sigma1_hat
# ) + pi2_hat * norm.pdf(x_sorted, mu2_hat, sigma2_hat)

# plt.plot(x_sorted, gaussian_kde(x_sorted)(x_sorted), color="g", linewidth=2)
# plt.plot(x_sorted, density_estimation, color="r", linewidth=2)
# plt.xlabel("X")
# plt.ylabel("Density")
# plt.title("Density estimation of X")
# plt.legend(["Kernel Density Estimation", "Maxture Density"])
# plt.show()

# ------------------------------------------------------------------------------

# ((((((((((((((( Understanding Hypothesis Testing )))))))))))))))

# ( Python Implementation of Case Using Critical values )

# import numpy as np
# from scipy import stats

# before_treatment = np.array([120, 122, 118, 130, 125, 128, 115, 121, 123, 119])
# after_treatment = np.array([115, 120, 112, 128, 122, 125, 110, 117, 119, 114])

# null_hypothesis = "The new drug has no effect on blood pressure"
# alternate_hypothesis = "The new drug has an effect on blood pressure"

# alpha = 0.05

# t_statistic, p_value = stats.ttest_rel(after_treatment, before_treatment)

# m = np.mean(after_treatment - before_treatment)
# s = np.std(after_treatment - before_treatment)
# n = len(before_treatment)
# t_statistic_manual = m / (s / np.sqrt(n))

# if p_value <= alpha:
#     decision = "Reject"
# else:
#     decision = "Fail to reject"

# if decision == "Reject":
#     conclusion = "There is statistically significant evidence that the average blood pressure before and after treatment with the new drug is different."
# else:
#     conclusion = "There is insufficient evidence to claim a significant difference in average blood pressure before and after treatment with the new drug."

# print(f"T-statistic: {t_statistic}")
# print(f"P-value: {p_value}")
# print(f"T-statistic (Calculated Manually): {t_statistic_manual}")
# print(f"Decision: {decision} the null hypothesis at alpha = {alpha}")
# print(f"Conclusion: {conclusion}")

# ------------------------------------------------------------------------------

# ( P-Value or probability value )
# a statistical measure used in hypothesis testing to assess the strength of evidence against a null hypothesis.

# import scipy.stats as stats

# sample_data = [78, 82, 88, 95, 79, 92, 85, 88, 75, 80]
# population_mean = 85

# t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

# print(f"T-statistic: {t_stat}")
# print(f"P-Value: {p_value}")

# aplha = 0.05
# if p_value < aplha:
#     print("Reject the null hypothesis. There is enough evidence to suggest a significant difference.")
# else:
#     print("Fail to reject the null hypothesis. The difference is not statistically significant.")

# ------------------------------------------------------------------------------

# ( Type II Error  )

# import math
# from scipy.stats import norm

# no_of_samples = 30
# pop_standard_dev = 120

# std_error = pop_standard_dev / math.sqrt(no_of_samples)

# print(std_error)

# alpha = 0.05
# m0 = 10000

# upper_bound = norm.ppf(alpha, loc=m0, scale=std_error)
# print(upper_bound)

# population_mean = 9950
# probability = 1 - norm.cdf(upper_bound, loc=population_mean, scale=std_error)
# print(probability)

# ------------------------------------------------------------------------------

# ( Confidence Interval )
# ( Using t-test )

# import scipy.stats as stats
# import math

# sample_mean = 240
# sample_std = 25
# sample_size = 10
# confidence_level = 0.95

# df = sample_size - 1
# alpha = (1 - confidence_level) / 2

# t_value = stats.t.ppf(1 - alpha, df)

# margin_of_error = t_value * (sample_std / math.sqrt(sample_size))

# lower_limit = sample_mean - margin_of_error
# upper_limit = sample_mean + margin_of_error

# print(f"Confidence Interval: ({lower_limit}, {upper_limit})")

# ------------------------------------------------------------------------------

# ( Confidence Interval )
# ( Using Z-test )

# import numpy as np

# sample_mean = 4.63
# std_dev = 0.54
# sample_size = 50
# confidence_level = 0.95

# standerd_error = std_dev / np.sqrt(sample_size)

# z_value = 1.960

# margin_of_error = z_value * (std_dev / np.sqrt(sample_size))

# lower_limit = sample_mean - margin_of_error
# upper_limit = sample_mean + margin_of_error

# print(f"Confidence Interval: ({lower_limit:.3f}, {upper_limit:.3f})")

# ------------------------------------------------------------------------------

# ((((((((((((((( Statistical Tests )))))))))))))))

# ( Z-test )
# Testing if the mean of a sample is significantly different from a known population mean

# ( One Sample Z test )

# import numpy as np
# from statsmodels.stats.weightstats import ztest

# data = [11.8] * 100
# population_mean = 12
# population_std_dev = 0.5

# z_statistic, p_value= ztest(data, value=population_mean)

# print(f"Z-statistic: {z_statistic:.4f}")
# print(f"P-value: {p_value:.4f}")

# alpha = 0.05
# if p_value < alpha:
#     print("Reject the null hypothesis: The average battery life is different from 12 hours.")
# else:
#     print("Fail to reject the null hypothesis: The average battery life is not significantly different from 12 hours.")

# ------------------------------------------------------------------------------

# ( Z-test --> Two Sample Z test )

# import numpy as np
# import scipy.stats as stats

# # Group A
# n1 = 50
# x1 = 75
# s1 = 10

# # Group B
# n2 = 60
# x2 = 80
# s2 = 12

# # Null Hypothesis = mu_1-mu_2 = 0
# # Hypothesized difference (under the null hypothesis)
# d = 0

# # set the segnificance level
# alpha = 0.05

# # Calculate the test statistic (z-score)
# z_score = ((x1 - x2) - d) / np.sqrt((s1**2 / n1) + (s2**2 / n2))
# print(f"Z-score: {abs(z_score)}")

# # Calculate the critical value
# z_critical = stats.norm.ppf(1 - alpha / 2)
# print(f"Critical score: {z_critical}")

# # Compare the test statistic with the critical value
# if abs(z_score) > z_critical:
#     print(
#         """Reject the null hypothesis.
# There is a significant difference b/w the online and offline classes."""
#     )
# else:
#     print(
#         """Fail to reject the null hypothesis.
# There is not evidence to suggest a significant difference b/w the online and offline classes."""
#     )

# # Approach 2: Using P-value

# # P-Value : Probability of getting less than a Z-score
# p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
# print(f"P-Value: {p_value}")


# # Compare the p-value with the significance level
# if p_value < alpha:
#     print(
#         """Reject the null hypothesis.
# There is a significant difference between the online and offline classes."""
#     )
# else:
#     print(
#         """Fail to reject the null hypothesis.
# There is not evidence to suggest significant difference b/w the online and offline classes."""
#     )

# ------------------------------------------------------------------------------

# ( T-test )
# Comparing means of two independent samples or testing if the mean of a sample is significantly different from a known or hypothesized population mean

# ( One sample T-test )

# import scipy.stats as stats
# import numpy as np

# population_mean = 45
# sample_mean = 75
# sample_std = 25
# sample_size = 25

# t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
# df = sample_size - 1
# alpha = 0.05
# critical_t = stats.t.ppf(1 - alpha, df)
# p_value = 1 - stats.t.cdf(t_statistic, df)

# print(f"T-Statistic: {t_statistic}")
# print(f"P-Value: {p_value}")
# print(f"Critical T-value: {critical_t}")

# print("with t-value:")
# if t_statistic > critical_t:
#     print(
#         """There is a significant difference in weight before and after the camp.
#     The fitness camp had an effect."""
#     )
# else:
#     print(
#         """There is no significant difference in weight before and after the camp.
#     The fitness camp did not have a significant effect."""
#     )

# print("With P-value :")
# if p_value > alpha:
#     print(
#         """There is a significant difference in weight before and after the camp.
#     The fitness camp had an effect."""
#     )
# else:
#     print(
#         """There is no significant difference in weight before and after the camp.
#     The fitness camp did not have a significant effect."""
#     )

# ------------------------------------------------------------------------------

#  ( T-test )
# ( Independent sample T-test )

# from scipy import stats
# import numpy as np

# sample_A = np.array(
#     [78, 84, 92, 88, 75, 80, 85, 90, 87, 7978, 84, 92, 88, 75, 80, 85, 90, 87, 79]
# )
# sample_B = np.array(
#     [82, 88, 75, 90, 78, 85, 88, 77, 92, 8082, 88, 75, 90, 78, 85, 88, 77, 92, 80]
# )

# t_statistic, p_value = stats.ttest_ind(sample_A, sample_B)

# alpha = 0.05
# df = len(sample_A) + len(sample_B) - 2

# critical_t = stats.t.ppf(1 - alpha / 2, df)

# print(f"T-value: {t_statistic}")
# print(f"P-value: {p_value}")
# print(f"Critical Value: {critical_t}")

# print("Wiht T-value")
# if abs(t_statistic) > critical_t:
#     print("There is significant difference between two groups")
# else:
#     print("No significant difference found between two groups")

# print("Wiht P-value")
# if p_value > alpha:
#     print(
#         "No evidence to reject the null hypothesis that a significant difference between the two groups"
#     )
# else:
#     print(
#         "Evidence found to reject the null hypothesis that a significant difference between the two groups"
#     )

# ------------------------------------------------------------------------------

#  ( T-test )
# ( Two-sample T-test )

# from scipy import stats
# import numpy as np

# math1 = np.array([4, 4, 7, 16, 20, 11, 13, 9, 11, 15])
# math2 = np.array([15, 16, 14, 14, 22, 22, 23, 18, 18, 19])

# t_statistic, p_vlaue = stats.ttest_rel(math1, math2)

# alpha = 0.05
# df = len(math2) - 1
# critical_t = stats.t.ppf(1 - alpha / 2, df)

# print(f"T-value: {t_statistic}")
# print(f"P-value: {p_vlaue}")
# print(f"Critical T-Value: {critical_t}")

# print("With T-value:")
# if abs(t_statistic) > critical_t:
#     print("There is significant difference between math1 and math2")
# else:
#     print("No significant difference found between math1 and math2")

# print("With P-value:")
# if p_vlaue > alpha:
#     print(
#         "No evidence to reject the null hypothesis that significant difference between math1 and math2"
#     )
# else:
#     print(
#         "Evidence found to reject the null hypothesis that significant difference between math1 and math2"
#     )

# ------------------------------------------------------------------------------

# ((((((((((((((( Chi-Squared Test )))))))))))))))
# The chi-squared test is a statistical test used to determine if there is a significant association between two categorical variables
# This test is also performed on big data with multiple number of observations.

# from scipy.stats import chi2_contingency

# # defining the table
# data = [[207, 282, 241], [234, 242, 232]]
# stat, p, pof, excepted = chi2_contingency(data)

# # interpret P-value
# alpha = 0.05
# print(f"P-value is: {str(p)}")
# if p <= alpha:
#     print("Dependent (reject H0)")
# else:
#     print("Independent (H0 holds true)")

# ------------------------------------------------------------------------------
# ((((((((((((((( Non-Parametric Test )))))))))))))))

# ( Mann-Whitney U Test )

# from scipy.stats import mannwhitneyu

# batch1 = [3, 4, 2, 6, 2, 5]
# batch2 = [9, 7, 5, 10, 8, 6]

# # perform mann whitney test
# stat, p_value = mannwhitneyu(batch1, batch2)
# print(f"Statistics = {stat:.2f}, p_value = {p_value:.2f}")

# alpha = 0.05
# if p_value < alpha:
#     print("Reject Null Hypothesis (Significant difference between two samples)")
# else:
#     print("Do not Reject Null Hypothesis (No significant difference between two samples)")
