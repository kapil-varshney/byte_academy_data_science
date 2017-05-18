# Part 1

oilprice = np.loadtxt('oil_price_monthly.dat', delimiter=',')
plt.plot(oilprice[:,2], 'b-')
nrow, ncol = oilprice.shape
months = ['January', 'February', 'March', 'April',\
          'May','June', 'July', 'August', 'September',\
		            'October', 'November', 'December']
					for price in [40, 60, 80]:
					    for i in range(nrow):
						        if oilprice[i, 2] > price:
								            print('The oil price exceeds ', price, 'euros for the first time in', \
											                  months[int(oilprice[i, 1])], 'of', int(oilprice[i, 0]))
															              break


# Part 2

from scipy.special import erf
def F(x, mu=0, sigma=1, p=0):
    rv = 0.5 * (1.0 + erf((x - mu) / np.sqrt(2 * sigma**2)))
	    return rv - p
		print('x=mu gives F(x)=', F(2, mu=2, sigma=1))
		print('x=mu+1.96sig gives:', F(2+1.96, mu=2, sigma=1))
		x1 = fsolve(F, 3, args=(3, 2, 0.1))
		x2 = fsolve(F, 3, args=(3, 2, 0.9))
		print('x1,F(x1):', x1, F(x1, mu=3, sigma=2))
		print('x2,F(x2):', x2, F(x2, mu=3, sigma=2))

# Part 3

from pandas import read_csv
w = read_csv('douglas_data.csv',skiprows=[1],skipinitialspace=True)
print('min and max bending strength: ', w.bstrength.min(), w.bstrength.max())
print('mean and std of density: ', w.density.mean(), w.density.std())
print('2.5%, 50%, 97.5% tree ring width: ', np.percentile(w.treering,[2.5,50,97.5]))

from pandas import read_csv
w = read_csv('douglas_data.csv',skiprows=[1],skipinitialspace=True)
print('2.5%, 50%, 97.5% tree ring width: ', w.treering.describe(percentiles=[0.025,0.5,0.975]))

