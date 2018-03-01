import pymc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
#Website A had 1055 clicks and 28 sign-ups
values_A = np.hstack(([0]*(1055-28),[1]*28))
#Website B had 1057 clicks and 45 sign-ups
values_B = np.hstack(([0]*(1057-45),[1]*45))
 
p_A = pymc.Uniform('p_A', 0, 1)
p_B = pymc.Uniform('p_B', 0, 1)
 
@pymc.deterministic
def delta(p_A = p_A, p_B = p_B):
    return p_B - p_A
 
obs_A = pymc.Bernoulli('obs_A', p_A, value = values_A , observed = True)
obs_B = pymc.Bernoulli('obs_B', p_B, value = values_B , observed = True)
model = pymc.Model([p_A, p_B, delta, values_A, values_B])
mcmc = pymc.MCMC(model)
 
mcmc.sample(1000000, 500000)
 
 
siteA_distribution = mcmc.trace("p_A")[:]
siteB_distribution = mcmc.trace("p_B")[:]
delta_distribution = mcmc.trace('delta')[:]
 
sns.kdeplot(delta_distribution, shade = True)
plt.axvline(0.00, color = 'black')
plt.savefig("results/2sites_diff.png", format = "PNG")
 
print "Probability that website A is MORE clicked on than site B: %0.3f" % (delta_distribution < 0).mean()
print "Probability that website A is LESS clicked on than site B: %0.3f" % (delta_distribution > 0).mean()