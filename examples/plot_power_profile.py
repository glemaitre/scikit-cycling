"""
======================================
Workflow to exploit cycling power data
======================================

This example illustrates the typical workflow to use to investigate cycling
power data.

"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

import matplotlib.pyplot as plt

from skcycling import Rider
# from skcycling.datasets import load_fit
from skcycling.datasets import load_rider

print(__doc__)

# rider = Rider(n_jobs=-1)
# rider.add_activities(load_fit())
rider = Rider.from_csv(load_rider())

rider.power_profile_.plot()
plt.xlabel('Time')
plt.ylabel('Power (W)')

rider.record_power_profile().plot(alpha=0.5, style='--', legend=True)
plt.xlabel('Time')
plt.ylabel('Power (W)')

plt.show()
