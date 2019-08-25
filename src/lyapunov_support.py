import numpy as np
from math import log
from src.nltsa_functions import psr


"""
	Baseado no Grassberger-Procaccia Algorithm
"""

def corr(series, s_std, n_points):
	"""
	Calcula a correlacao
	:param serie: serie temporal
	:param s_std: desvio padrao da serie
	:param n_points: quantidade de pontos considerada
	:returns: correlacao entre todas as subseries
	"""
	somatorio=0
	for i in range(n_points):
		for j in range(n_points):
			ti, tj = [], []			
			for k in range(len(series)):
				ti.append(series[k][i])
				tj.append(series[k][j])
				
			#            heaviside_step
			somatorio += int(s_std - np.linalg.norm(np.array(ti) - np.array(tj)) > 0)

	return ( 2 / float(n_points * (n_points - 1))) + somatorio

def find_saturation(dimensions, log_corr):
	"""
	Busca a primeira dimensao onde o ganho na correacao e pequeno
	:param serie: serie temporal
	:param s_std: desvio padrao da serie
	:param n_points: quantidade de pontos considerada
	:returns: correlacao da subserie
	"""
	if len(log_corr) != len(dimensions): raise ValueError("Lists must have same size")
	if len(log_corr) < 2: raise ValueError("List must be larger then 2")

	old = np.array(log_corr)
	new = np.roll(old, -1)
	comp = (abs(old - new) <= abs(new / 100))[:-1]

	if sum(comp) == 0: return None
	else: return dimensions[np.argmax(comp)]

def get_dim(serie, tau, dmin=1, dmax=20, n_points=10):
	"""
	Calcula a dimensao
	:param serie: uma serie temporal
	:param tau: tamanho do lag
	:param mmin: menor dimensao a ser testada
	:param mmax: maior dimensao a ser testada
	:param n_points: quantidade de pontos analisados na serie
	:returns: dimensao encontrada ou None caso nao encontre
	"""
	s_std = np.std(serie[:n_points])
	dimensions = np.arange(dmin, dmax + 1)
	log_corr = [log(corr(psr(serie, m, tau), s_std, n_points), s_std) for m in dimensions]
	return find_saturation(dimensions, log_corr)

from scipy import ndimage
EPS = np.finfo(float).eps

def mutual_information(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi

def find_lag(serie, max_tau=3000):
	'''
	 Fraser and Swinney [1] suggest using the first local minimum of the mutual information 
	 between the delayed and non-delayed time series, effectively identifying a value of 
	 τ for which they share the least information.

	[1] Andrew M. Fraser and Harry L. Swinney, Independent coordinates for strange attractors 
	from mutual information - Phys. Rev. A 33, 1134 – Published 1 February 1986I
	fonte: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.33.1134

	:param serie: serie temporal
	:param max_tau: maximo de iteracoes
	'''
	mis = np.array([mutual_information(serie[:-tau], np.roll(serie, -tau)[:-tau], normalized=True) 
					for tau in range(1, max_tau)])

	# first min
	return np.argmax(mis - np.roll(mis, -1) < 0) + 1