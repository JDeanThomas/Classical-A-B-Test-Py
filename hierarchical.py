import pymc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@pymc.stochastic(dtype=np.float64)
def beta_priors(value=[1.0, 1.0]):
    a, b = value
    if a <= 0 or b <= 0:
        return -np.inf
    else:
        return np.log(np.power((a + b), -2.5))

a = beta_priors[0]
b = beta_priors[1]

#The hidden, true rate for each website.
true_rates = pymc.Beta('true_rates', a, b, size=5)

# This is what we observed
trials = np.array([1055, 1057, 1065, 1039, 1046])
successes = np.array([28, 45, 69, 58, 60])
observed_values = pymc.Binomial('observed_values', trials, true_rates, observed=True, value=successes)

model = pymc.Model([a, b, true_rates, observed_values])
mcmc = pymc.MCMC(model)

# Generate 1M samples, and throw out the first 500k
mcmc.sample(1000000, 500000)


diff_CA = mcmc.trace('true_rates')[:][:,2] - mcmc.trace('true_rates')[:][:,0]
sns.kdeplot(diff_CA, shade = True, label = "Difference site C - site A")
plt.axvline(0.0, color = 'black')
plt.savefig("results/hierarchial_CA.png", format = "PNG")

print "Probability that website A is MORE clicked on than website C: %0.3f" % (diff_CA < 0).mean()
print "Probability that website A is LESS clicked on than website C: %0.3f" % (diff_CA > 0).mean()