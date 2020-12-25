import numpy as np, pandas as pd, math, time, random, multiprocessing, copy, logging
from collections import Counter
from functools import partial
from scipy.stats import chisquare
from scipy import stats, integrate
import argparse; from argparse import ArgumentParser
parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-rp',          type = int,        default = 5,         help = "Repeat for each parameter space")
parser.add_argument('-np',          type = int,        default = 5,         help = "The number of processes")

parser.add_argument('-T',           type = float,      default = 1e3,       help = "Generation to which it will simulate")
parser.add_argument('-step',        type = int,        default = 100,       help = "Time step for recording")

parser.add_argument('-N',           type = int,        default = 100,       help = "Constant population size in WF model")
parser.add_argument('-u',           type = float,      default = 0.6,       help = "Initial mutation rate per reproduction (per haploid or diploid genome) without mutators")
parser.add_argument('-hr',          type = float,      default = 0.5,       help = "Homologous recombination rate. This option is for diploid sexual population. Not necessary for haploid population. range from 0 (no recombination) to 0.5 (free recombination)")
parser.add_argument('-pld',         type = int,         default = 1,        help = "Ploidy. It must be 1 or 2. 1 for asexual haploid population, 2 for sexual diploid population")

parser.add_argument('-r',           type = float,      default = 0.00,      help = "Proportion to be mutators")
parser.add_argument('-p',           type = float,      default = 0.00,      help = "Proportion to be positive mutations")
parser.add_argument('-q',           type = float,      default = 1.00,      help = "Proportion to be negative mutations")
parser.add_argument('-ld',          type = float,      default = 1.00,      help = "average factor increasing mutation rate. i.e. u'=ld*u")
parser.add_argument('-sp',          type = float,      default = 0.05,      help = "selection coefficient of positive mutations, i.e. fitness gain, should be s > 0")
parser.add_argument('-sq',          type = float,      default = 0.01,      help = "selection coefficient of negative mutations, i.e. fitness loss, should be s > 0")
parser.add_argument('-ld_isConst',  type = bool,       default = True,      help = "Whether the factor of mutator strength is a constant value")
parser.add_argument('-sp_isConst',  type = bool,       default = True,      help = "Whether positive selection coeffcient is a constant value")
parser.add_argument('-sq_isConst',  type = bool,       default = True,      help = "Whether negative selection coeffcient is a constant value")
args = parser.parse_args()


def fixP(N, s, p, ploidy): 
    '''
    fixation probability of a mutant (whose selection coeffcient is s) with frequency p.
    This can apply to both haploid (ploidy = 1) and diploid (ploidy = 2) populations. 
    In diploid population, we consider the genic selection (also h=0.5, 2s for homozygote, hs=s for heterozygote).
    s>0 for advantageous mutations, s<0 for deleterious mutations. s=0 for neutral mutations.
    Note: N is population size, rather than the number of alleles
    '''
    assert ploidy == 1 or ploidy == 2
    c = 2 * N * s * ploidy
    e = math.e
    try:
        f = p if s == 0 else (1 - e ** (-c * p)) / (1 - e ** (-c))
    except OverflowError: ## when negative selection coeffcient (s<0) is very small, it will overflow.
        f = 0
    return f

def absorbT(N, s, p, ploidy):
    '''
    Obtain average fixation time and extinct time by numerical integration. More see Crow and Kimura (P430). #TODO: test haploid
    not correct when s<0.
    '''
    def M(x): return s*x*(1-x)
    def V(x): return x*(1-x)/(ploidy * N)
    def G(x): return math.exp(-2*ploidy*N*s*x)
    def u(x): return fixP(N, s, x, ploidy)
    def Psi(x): 
        res, err = integrate.quad(G, 0, 1)
        return 2 * res / V(x) / G(x)
    def fp1_t1(x): 
        return Psi(x) * u(x) * (1-u(x))
    def f0p_t1(x):
        return Psi(x) * u(x) ** 2
    def fp1_t0(x): 
        return Psi(x) * (1-u(x)) ** 2
    def f0p_t0(x):
        return Psi(x) * (1-u(x)) * u(x)
    t1 = integrate.quad(fp1_t1, p, 1)[0] + (1-u(p))/u(p) * integrate.quad(f0p_t1, 0, p)[0]  ## average fixation time
    t0 = u(p)/(1-u(p)) * integrate.quad(fp1_t0, p, 1)[0] + integrate.quad(f0p_t0, 0, p)[0]  ## average extinct time
    return t1, t0
    
def absorbT_neutral(N, p, ploidy):
    '''Obtain average fixation time and extinct time for a neutral mutation by analytical equation. More see Crow and Kimura. #TODO: test haploid'''
    c = ploidy * 2 * N
    q = 1-p
    t1_p = -(c * q * math.log(q)) / p   ## average fixation time
    t0_p = -c * (p / q) * math.log(p)   ## average extinct time
    return t1_p, t0_p

def traceLineage(mlineage, recentMut): 
    """
    A function to obtain the mutational lineage of an individual from the mutation id of the most recently occurred mutation in the cell. 
    We use the tree data structure to record the mutations. For example, the input ID (most recently occurred mutation) of target cell is "100" and the output is [100, 56, 12, 1], which is the mutation lineage of the cell
    ::mlineage::  list   -- a tree data structure that could be used to trace the mutational lineage given the most recent mutation id of a lineage, the element is a string-format integer, e.g. '1' or '2,3'
    ::recentMut:: string -- the mutation ID of the most recently occurred mutation in the cell, BCF: if the recentMut is '0', the function will return a null list []
    ::return::    list   -- list of integer like [100, 56, 12, 1], BCF: the return element is integer type rather than string type, not containing the initial mutation id (0)
    """
    recent_muts = recentMut.split(',')  # it is possible that multiple mutations occur during in a cell division, e.g. "100,101"
    recent_muts = [int(t) for t in sorted(recent_muts, reverse = True)]  # the order is descreasing 101, 100
    first_mut = recent_muts[-1]
    trace = []
    while first_mut > 0:
        trace.extend(recent_muts)
        recent_muts = mlineage[first_mut].split(',')
        recent_muts = [int(t) for t in sorted(recent_muts, reverse = True)]
        first_mut = recent_muts[-1]
    return trace  
 
def mutRandom(totMut = 100, r = 0.05, p = 0.03, q = 0.17, ld = 1.1, sp = 0.05, sq = 0.01, ld_isConst = True, sp_isConst = True, sq_isConst = True):
    '''
    Random mutations in a given number.
    totMut:positive int -- the maximal mutation number at which simulation will stop
    r : positive float -- probability to be mutator mutation
    p : positive float -- probability to be positive mutations
    q : positive float -- probability to be negative mutations
    ld: positive float -- strength of each mutator
    sp: positive float -- mean selection coefficient of positive mutations (i.e. fitness gain), the lambda = 1/sp for exponential distribution
    sq: positive float -- mean selection coefficient of negative mutations (i.e. fitness loss), the lambda = 1/sq for exponential distribution
    ld_isConst: bool   -- decide whether ld is a contant or not. If not a contant, ld will be sampled from log exponential with mean approximately equal to ld
    sp_isConst: bool   -- decide whether sp is a contant or not. If not a contant, selection coeffcient will be sampled from exponential with mean equal to sp
    sq_isConst: bool   -- decide whether sq is a contant or not. If not a contant, selection coeffcient will be sampled from exponential with mean equal to sq, and then obtain its opposite number
    :: return ::
    mutDF: pd DataFrame -- 4 ('r', 'p', 'q', 'n') by totMut matrix. Each column is a mutation and it has only one non-NaN number.
    '''
    m = np.log(ld)
    r_lamb = np.repeat(ld, totMut) if ld_isConst else np.exp(np.random.exponential(m, size = totMut))                   # sample the mutator effect factor (lambda)
    p_sCoe = np.repeat(sp, totMut) if sp_isConst else np.random.exponential(sp, size = totMut)                          # sample positive selection coefficients
    q_sCoe = np.repeat(-sq, totMut) if sq_isConst else -np.random.exponential(sq, size = totMut)                        # sample negative selection coefficients, BCF: transform to negative float (its opposite number)
    n_sCoe = np.zeros(totMut)                                               # sample neutral selection coefficients (0)
    valMatrix = pd.DataFrame(np.array([r_lamb, p_sCoe, q_sCoe, n_sCoe]), index = list('rpqn'))  
    
    mutPs = np.array([r, p, q, 1-r-p-q])
    indexL = np.random.choice([0, 1, 2, 3], size = totMut, p = mutPs)
    indexMatrix = np.zeros((4, totMut))
    for c, r in enumerate(indexL):
        indexMatrix[(r,c)] = 1
    df = np.array(valMatrix)
    df[indexMatrix == 0] = np.nan
    mutDF = pd.DataFrame(df, index = list('rpqn'), columns = range(1, df.shape[1] + 1))
    return mutDF   
    
def mutDF_toDict(mutDF, initID):
    mut_num = mutDF.shape[1]
    mut_id_list = range(initID, initID + mut_num)
    preSer = mutDF.apply(lambda x:list(x.dropna().items())[0], axis = 0)
    preSer.index = mut_id_list
    outDict = dict(preSer)
    return outDict
    
class Individual_Diploid():
    '''Individual class for diploid sexual (hermaphroditic) population'''
    def __init__(self, u = 0.3, w = 1, chrA = [], chrB = []):
        '''
        u       : float     -- mutation rate per reproduction (per diploid genome)
        w       : float     -- fitness
        chrA    : list      -- list of mutation id (positive integer) in one of the homologous chromosomes 
        chrB    : list      -- list of mutation id (positive integer) in the other homologous chromosomes 
        '''
        self.u, self.w = u, w  
        self.chrA, self.chrB = copy.deepcopy(chrA), copy.deepcopy(chrB)
    
class Population_Diploid(dict):
    '''Population class for diploid sexual (hermaphroditic) population'''
    initW = 1
    def __init__(self, indvdL, mutDict, initT = 0, u0 = 0.3, recombinationR = 0.5, randomMutFunc = mutRandom):
        '''
        indvdL          : list      -- a list of Individual_Diploid instance
        mutDict         : dict      -- dict to record the mutations in population. keys: mutID (positive integer), values: (mutType, mutInfo). e.g. {1: ('p', 0.2), 2: ('q', -0.1), 3: ('n', 0), 7: ('r', 1.03)}. Note that this dict maybe record some of extinct mutations. We can clear them out by self.clearMutDict() after some time to save memory.
        initT           : int       -- the initial time (i.e. generation)
        u0              : float     -- the initial mutation rate per reproduction without any mutators. If u0=0, don't consider new muations
        recombinationR  : float     -- the recombination rate, from 0 - 0.5. Although it can range from 0 to 1, 0.1 is the effect as 0.9.
        randomMutFunc   : function  -- a function to generate a sequence of different type of mutations (more see the default function). In fact, we can decide which mutation it is each time a mutation occurs. But this method slows down the simulation speed. Thus, we generate numerous mutations in one time. 
        '''        
        indvdL, mutDict = copy.deepcopy(indvdL), copy.deepcopy(mutDict)  ##  a mutable object need to be recreated to avoid unforeseen consequences
        self.mutDict, self.randomMut = mutDict, randomMutFunc
        self.size, self.time, self.u0, self.r = len(indvdL), initT, u0, recombinationR
        mutID = 0  ## the maximal id of current mutations
        for id, init_indvd in enumerate(indvdL, 1):  
            indvd = self.indvd_update(init_indvd) ## update the attributes of individuals according to its carried mutations
            self[id] = indvd
            mutID = max(indvd.chrA + indvd.chrB + [mutID])
        self.mutID = mutID  ## the maximal id of current mutations
    def alleleFitness(self, allele):
        '''Get the fitness of the input allele. Allele is just a list of mutations, e.g. the chrA or chrB of individual'''
        w = Population_Diploid.initW
        mutID_list = allele
        for mutID in mutID_list:
            mutType, mutInfo = self.mutDict[mutID]
            if mutType == 'q':          # negative effect
                # w /= (1 - mutInfo)
                w *= (1 + mutInfo)
            elif mutType == 'n': pass   # neutral effect
            elif mutType == 'p':        # positive effect
                w *= (1 + mutInfo)   
            else: pass                  # mutator effect
        return w
    def indvd_update(self, indvd):
        u, w = self.u0, Population_Diploid.initW
        mutID_list = indvd.chrA + indvd.chrB
        for mutID in mutID_list:
            mutType, mutInfo = self.mutDict[mutID]
            if mutType == 'q':          # negative effect
                # w /= (1 - mutInfo)
                w *= (1 + mutInfo)
            elif mutType == 'n': pass   # neutral effect
            elif mutType == 'p':        # positive effect
                w *= (1 + mutInfo)
            else: u *= mutInfo          # mutator effect
        indvd.u, indvd.w = u, w
        return indvd
    def sfs(self):
        sitePool = []
        for indvd in self.values():
            sitePool.extend(indvd.chrA)
            sitePool.extend(indvd.chrB)
        site_count = Counter(sitePool)
        return site_count
    def clearMutDict(self):
        '''Clear out extinct mutations and only keep the mutations in current population. We can do this to save memory after some time'''
        site_count = self.sfs()
        cleanDict = {id: self.mutDict[id] for id in site_count.keys()}
        self.mutDict = cleanDict
    def indvdReproduce(self, indvd):
        '''
        Individual reproduction: accumulate mutations first, and then homologous recombination, two gametes are finally segregated. 
        return  : two list of mutID for two gametes
        '''
        #### mutation
        chrA_mut_num, chrB_mut_num = np.random.poisson(lam = indvd.u/2, size = 2)
        tot_mut_num = chrA_mut_num + chrB_mut_num
        if not tot_mut_num:  ## no new mutations
            pass
        else:
            indvd.chrA.extend(range(self.mutID+1, self.mutID+1+chrA_mut_num))
            indvd.chrB.extend(range(self.mutID+1+chrA_mut_num, self.mutID+1+tot_mut_num))
            mutDF = self.randomMut(tot_mut_num)
            newMut_dict = mutDF_toDict(mutDF, initID = self.mutID+1); self.mutDict.update(newMut_dict)
            self.mutID += tot_mut_num
        #### recombination
        mutA_set, mutB_set = set(indvd.chrA), set(indvd.chrB)
        homo_mutL, mutA_private, mutB_private = mutA_set.intersection(mutB_set), mutA_set - mutB_set, mutB_set - mutA_set
        newChrA, newChrB = list(homo_mutL), list(homo_mutL)     ## haplotype after recombination
        for mut in mutA_private:
            if np.random.binomial(1, self.r): ## homologous recombination(i.e. to chrB)
                newChrB.append(mut)
            else: newChrA.append(mut)
        for mut in mutB_private:
            if np.random.binomial(1, self.r): ## homologous recombination (i.e to chrA)
                newChrA.append(mut)
            else: newChrB.append(mut)
        newChrA.sort(); newChrB.sort()        ## sort the mutID
        return newChrA, newChrB
    def WrightFisher_stepForward(self):
        '''selection-reproduction(mutation and recombination)'''
        self.time += 1
        ##### selection
        idL = list(self.keys())
        wL = [indvd.w for indvd in self.values()]
        MaleIDs = random.choices(idL, weights = wL, k = self.size)    ## sample with replacement
        FemaleIDs = random.choices(idL, weights = wL, k = self.size)  ## sample with replacement
        Male_indvds = [copy.deepcopy(self[id]) for id in MaleIDs]     ## need the deepcopy, because Wright-Fisher model maybe pick the same individual several times
        Female_indvds = [copy.deepcopy(self[id]) for id in FemaleIDs]     ## need the deepcopy, because Wright-Fisher model maybe pick the same individual several times
        self.clear()
        ##### reproduction (mutation and recombination)
        for indvdM, indvdF, id in zip(Male_indvds, Female_indvds, idL): ## don't change id
            MaleChr = random.choice(self.indvdReproduce(indvdM))
            FemaleChr = random.choice(self.indvdReproduce(indvdF))
            offspring = Individual_Diploid(chrA = MaleChr, chrB = FemaleChr)
            offspring_update = self.indvd_update(offspring)  ## update w and u after combination of two gametes 
            self[id] = offspring_update

def simToAbsorb_diploid(N, s, freq = 1, r = 0.5):
    '''
    Simulate dynamics of a mutant until its absorbtion (fixation or extinction) in diploid population. Note freq should not be greater than N
    s   : float -- selection coeffcient, s=0 for neutral, s>0 for positive, s<0 for negative.
    freq: positive integer -- initial number (freq) of mutant 
    return: tuple -- isFix (1 or 0), absorbT: time to absorbtion (fixation or extinction)
    '''
    target_mutID = 1
    indvds_with_targetMut = [Individual_Diploid(u=0, w=1+s, chrA = [target_mutID], chrB = []) for i in range(freq)]
    indvds_without_targetMut = [Individual_Diploid(u=0, w=1, chrA = [], chrB = []) for i in range(0, N-freq)]
    indvdL = indvds_with_targetMut + indvds_without_targetMut
    if s == 0: mutType = 'n'
    elif s > 0: mutType = 'p'
    else: mutType = 'q'
    mutDict = {target_mutID: (mutType, s)}
    p = Population_Diploid(indvdL, mutDict, u0 = 0, recombinationR = r)   # # u0 = 0, no new mutations
    target_count = p.sfs()[target_mutID]
    fixCount = 2 * N
    while target_count != 0 and target_count != fixCount:
        p.WrightFisher_stepForward()
        target_count = p.sfs()[target_mutID]
    absorbT = p.time
    isFix = 1 if target_count == fixCount else 0
    return isFix, absorbT

def repeatAbsorb_diploid(repeat, N, s, freq = 1, r = 0.5):
    fixCase, fixT_list, extinctT_list = 0, [], []
    for i in range(repeat):
        isFix, absorbT = simToAbsorb_diploid(N, s, freq, r)
        if isFix:
            fixCase += 1
            fixT_list.append(absorbT)
        else:
            extinctT_list.append(absorbT)
    fixP = fixCase / repeat
    return fixP, fixT_list, extinctT_list

class Individual_Haploid():
    '''Individual class for haploid asexual (cell) population. This also can be applied to cell population even though the ploidy of cells is 2.'''
    def __init__(self, u = 0.3, w = 1, chrH=[]):
        '''
        u       : float     -- mutation rate per reproduction (per diploid/haploid genome)
        w       : float     -- fitness
        chrH    : list      -- list of mutation id (positive integer) in haploid genome 
        '''
        self.u, self.w = u, w
        self.chrH = copy.deepcopy(chrH)

class Population_Haploid(dict):
    '''Population class for haploid asexual (cell) population. This also can be applied to cell population even though the ploidy of cells is 2.'''
    initW = 1
    def __init__(self, indvdL, mutDict, initT = 0, u0 = 0.3, randomMutFunc = mutRandom):
        '''
        indvdL          : list      -- a list of Individual_Haploid instance
        mutDict         : dict      -- dict to record the mutations in population. keys: mutID (positive integer), values: (mutType, mutInfo). e.g. {1: ('p', 0.2), 2: ('q', -0.1), 3: ('n', 0), 7: ('r', 1.03)}. Note that this dict maybe record some of extinct mutations. We can clear them out by self.clearMutDict() after some time to save memory.
        initT           : int       -- the initial time (i.e. generation)
        u0              : float     -- the initial mutation rate per reproduction without any mutators. If u0=0, don't consider new muations
        randomMutFunc   : function  -- a function to generate a sequence of different type of mutations (more see the default function). In fact, we can decide which mutation it is each time a mutation occurs. But this method slows down the simulation speed. Thus, we generate numerous mutations in one time. 
        '''        
        indvdL, mutDict = copy.deepcopy(indvdL), copy.deepcopy(mutDict)  ##  a mutable object need to be recreated to avoid unforeseen consequences
        self.mutDict, self.randomMut = mutDict, randomMutFunc
        self.fixMutDict = {}
        self.size, self.time, self.u0 = len(indvdL), initT, u0
        mutID = 0  ## the maximal id of current mutations
        self.fixMut_update()
        for id, init_indvd in enumerate(indvdL, 1):  
            indvd = self.indvd_update(init_indvd) ## update the attributes of individuals according to its carried mutations
            self[id] = indvd
            mutID = max(indvd.chrH + [mutID])
        self.mutID = mutID  ## the maximal id of current mutations
    def fixMut_update(self):
        u, w = self.u0, Population_Haploid.initW 
        nMut, qMut, pMut, rMut = 0, 0, 0, 0  ## the fixed number of neutral, negative, positive, mutator mutations
        for mutID in self.fixMutDict:
            mutType, mutInfo = self.fixMutDict[mutID]
            if mutType == 'q':              # negative effect
                # w /= (1 - mutInfo)
                w *= (1 + mutInfo)
                qMut += 1
            elif mutType == 'n': nMut+=1    # neutral effect
            elif mutType == 'p':            # positive effect
                w *= (1 + mutInfo)
                pMut += 1
            else:                           # mutator effect
                u *= mutInfo  
                rMut += 1
        self.fixU, self.fixW, self.nMut, self.qMut, self.pMut, self.rMut = u, w, nMut, qMut, pMut, rMut
    def indvd_update_old(self, indvd):
        u, w = self.u0, Population_Haploid.initW
        for mutID in indvd.chrH:
            mutType, mutInfo = self.mutDict[mutID]
            if mutType == 'q':          # negative effect
                # w /= (1 - mutInfo)
                w *= (1 + mutInfo)
            elif mutType == 'n': pass   # neutral effect
            elif mutType == 'p':        # positive effect
                w *= (1 + mutInfo)
            else: u *= mutInfo          # mutator effect
        indvd.u, indvd.w = u, w
        return indvd
    def indvd_update(self, indvd):
        # self.fixMut_update()  ## update the fix attributes to update self.fixU
        u, w = self.fixU, Population_Haploid.initW  ## the baseline mutation rate is supposed to be based on fixed mutator mutations
        new_chrH = [mutID for mutID in indvd.chrH if mutID not in self.fixMutDict]
        indvd.chrH = new_chrH
        for mutID in indvd.chrH:
            mutType, mutInfo = self.mutDict[mutID]
            if mutType == 'q':          # negative effect
                # w /= (1 - mutInfo)
                w *= (1 + mutInfo)
            elif mutType == 'n': pass   # neutral effect
            elif mutType == 'p':        # positive effect
                w *= (1 + mutInfo)
            else: u *= mutInfo          # mutator effect
        indvd.u, indvd.w = u, w
        return indvd
    def updateAll(self):
        self.clearMutDict()  ## clear out extinct mutation, add new fixed mutation. But chrH still include fixed mutations
        # self.fixMutDict()  ## update fixation attributes according to fixmation mutations, have been done in above command
        for id, indvd in self.items():  ## update each individual: remove the fixed mutations and update its u ans w
            self.indvd_update(indvd) 
    def constructSeq(self, alterMarker = 'A', controlMarker = 'C'):
        '''Construct a sequence for each individual in current population'''
        pedigreeDict = {id: indvd.chrH for id, indvd in self.items()}
        siteSet = set()
        for id, lineage in pedigreeDict.items():
            siteSet = siteSet.union(set(lineage))
        increaseSiteList = sorted(list(siteSet))
        seqDict = {}
        for id, lineage in pedigreeDict.items():
            seq = ''
            for site in siteSet:
                if site in lineage: seq += alterMarker # alteration
                else: seq += controlMarker  # ref
            seqDict[id] = seq
        return seqDict
    def toMega(self, outName = 'toMega.fasta', alterMarker = 'A', controlMarker = 'C'):
        '''Construct the sequence for each individual in current population and save for Mega software to construct tree'''
        seqDict = self.constructSeq(alterMarker, controlMarker)
        site_count = self.sfs()
        totMut = len(site_count)
        fixMut = Counter(site_count.values())[self.size]
        with open(outName, 'w') as f:
            f.write('>control (%sbp) | fixMut: %s\n' %(totMut, fixMut)) ## construct a control seq
            f.write(controlMarker * totMut + '\n')
            for i, seq in enumerate(seqDict.values(), 1):
                iMut = seq.count(alterMarker)  ## the mutation number of the i-th seq
                f.write('>seq%s (%s)\n' %(i, iMut))
                f.write(seq + '\n')
    def afs(self): 
        '''
        obtain the allele frequency spectrum according to the infinite allele model. Because there is no recombination, we can simply use recentMut of an individual to represnt its haplotype (or allele). Note: If there is no mutation (i.e. len(indvd.chrH)=0), just let 0 represents the allele without mutation.
        :: return :: dict   -- key is a unique allele represented by recentMut (integer), value is the frequency of this allele. e.g. Counter({1: 2, 2: 1, 10: 1}): there are 3 alleles with frequencies 2, 1, 1 respectively.
        '''
        allele_pool = [indvd.chrH[-1] if len(indvd.chrH)>0 else 0 for indvd in self.values()] 
        allele_count = Counter(allele_pool)
        return allele_count
    def sfs(self):
        ''''''
        sitePool = []
        for indvd in self.values():
            sitePool.extend(indvd.chrH)
        site_count = Counter(sitePool)
        return site_count
    def clearMutDict_old(self):
        '''Clear out extinct mutations and only keep the mutations in current population. We can do this to save memory after some time'''
        site_count = self.sfs()
        cleanDict = {id: self.mutDict[id] for id in site_count.keys()}
        self.mutDict = cleanDict
    def clearMutDict(self):
        '''Clear out extinct mutations and only keep the mutations in current population. We can do this to save memory after some time. Moreover, we only trace the polymorphic sites and remove the fixed mutations'''
        site_count = self.sfs()  
        new_fixMutID_list = [k for k, v in site_count.items() if v==self.size]  ## new fixation mutations
        new_fixMutDict = {id: self.mutDict[id] for id in new_fixMutID_list}
        snpMutID_list = [k for k, v in site_count.items() if v <self.size]      ## snps in extinct individuals
        snpMutDict = {id: self.mutDict[id] for id in snpMutID_list}
        self.fixMutDict.update(new_fixMutDict)   ## update the fixMutDict
        self.fixMut_update()                     ## update according to the new fixed mutations
        self.mutDict = snpMutDict
    def indvdReproduce(self, indvd):
        '''
        Individual reproduction: accumulate mutations first, and then update its attributes (u, w) according to new mutations. Because there is no recombination here, we don't need to update its attributes redundantly as diploid population does.
        return  : an Individual_Haploid instance with mutation accumulation and updated attributes
        '''
        ##### mutations
        mut_num = np.random.poisson(lam = indvd.u, size = None) ## if no new mutation, it takes 95% time. else it only take 1% time
        if not mut_num:   ## no new mutations
            return indvd
        else:
            indvd.chrH.extend(range(self.mutID+1, self.mutID+1+mut_num))
            mutDF = self.randomMut(mut_num)                             ## take 41% time
            newMut_dict = mutDF_toDict(mutDF, initID = self.mutID+1)    ## take 57% time
            self.mutDict.update(newMut_dict)
            self.mutID += mut_num
            # indvd = self.indvd_update(indvd) ## okay, but don't need to update the attributes redundantly
            for mutType, mutInfo in newMut_dict.values():
                if mutType == 'q':                  # negative effect
                    # indvd.w /= (1 - mutInfo)
                    indvd.w *= (1 + mutInfo)
                elif mutType == 'n': pass           # neutral effect
                elif mutType == 'p':                # positive effect
                    indvd.w *= (1 + mutInfo)
                else: indvd.u *= mutInfo            # mutator effect
            return indvd           
    def WrightFisher_stepForward(self):
        '''selection-reproduction(mutation)'''
        self.time += 1
        ##### selection 
        idL = list(self.keys())
        wL = [indvd.w for indvd in self.values()]
        sampleIDs = random.choices(idL, weights = wL, k = self.size)        ## sample with replacement
        reproduce_indvds = [copy.deepcopy(self[id]) for id in sampleIDs]    ## need the deepcopy, because Wright-Fisher model maybe pick the same individual several times  ## 35% time
        self.clear()
        ##### reproduction (mutation)
        for id, indvd in zip(idL, reproduce_indvds):
            offspring = self.indvdReproduce(indvd)       ## 60% time
            # offspring = self.indvd_update(offspring)   ## okay, but don't need to update the attributes redundantly
            self[id] = offspring

def simToAbsorb_haploid(N, s, freq = 1):
    '''
    Simulate dynamics of a mutant until its absorbtion (fixation or extinction) in haploid population. Note freq should not be greater than N
    s   : float -- selection coeffcient, s=0 for neutral, s>0 for positive, s<0 for negative.
    freq: positive integer -- initial number (freq) of mutant 
    return: tuple -- isFix (1 or 0), absorbT: time to absorbtion (fixation or extinction)
    '''
    target_mutID = 1
    indvds_with_targetMut = [Individual_Haploid(u=0, w=1+s, chrH = [target_mutID]) for i in range(freq)]
    indvds_without_targetMut = [Individual_Haploid(u=0, w=1, chrH=[]) for i in range(0, N-freq)]
    indvdL = indvds_with_targetMut + indvds_without_targetMut
    if s == 0: mutType = 'n'
    elif s > 0: mutType = 'p'
    else: mutType = 'q'
    mutDict = {target_mutID: (mutType, s)}
    p = Population_Haploid(indvdL, mutDict, u0 = 0)
    target_count = p.sfs()[target_mutID]
    fixCount = N
    while target_count != 0 and target_count != fixCount:
        p.WrightFisher_stepForward()
        target_count = p.sfs()[target_mutID]
    absorbT = p.time
    isFix = 1 if target_count == fixCount else 0
    return isFix, absorbT

def repeatAbsorb_haploid(repeat, N, s, freq = 1):
    fixCase, fixT_list, extinctT_list = 0, [], []
    for i in range(repeat):
        isFix, absorbT = simToAbsorb_haploid(N, s, freq)
        if isFix:
            fixCase += 1
            fixT_list.append(absorbT)
        else:
            extinctT_list.append(absorbT)
    fixP = fixCase / repeat
    return fixP, fixT_list, extinctT_list


def wfSimToT_diploid(T, tStep, N, u, r, p, q, ld, sp, sq, ld_isConst, sp_isConst, sq_isConst, hr, **kwargs):
    '''hr: homologous recombination rate'''
    ###### step 1: set up initial population
    kwargs_mutRandom = dict(r = r, p = p, q = q, ld = ld, sp = sp, sq = sq, ld_isConst = ld_isConst, sp_isConst = sp_isConst, sq_isConst = sq_isConst)
    randomMutFunc = partial(mutRandom, **kwargs_mutRandom)
    init_indvdL = [Individual_Diploid(u = u, w = 1, chrA = [], chrB = []) for i in range(N)]
    mutDict = dict()
    wfPop = Population_Diploid(indvdL = init_indvdL, mutDict = mutDict, initT = 0, u0 = u, recombinationR = hr, randomMutFunc = randomMutFunc)
    ###### step 2: trace dynamics by time
    tL = []
    wminL, wmaxL, wmeanL = [], [], []
    loadMinL, loadMaxL, loadMeanL = [], [], []
    totNumL, fixNumL = [], []
    random.seed(); np.random.seed() ##### Because all the child processes are forked from the same main process and then np.random.seed (random.seed does not, but also add here) will produce the same values in different child processes. Need to add this command for parallel processes. 
    fixCount = 2 * N
    while wfPop.time <= T:
        if wfPop.time % tStep == 0: ## only record at some time points
            site_count = wfPop.sfs()
            cleanDict = {id: wfPop.mutDict[id] for id in site_count.keys()}; wfPop.mutDict = cleanDict ## clear out extinct mutations to save memory
            # wfPop.clearMutDict()    ## the same effect as the above command, don't use this because it will use the function self.sfs() redundantly
            # wL, loadL = [indvd.w for indvd in wfPop.values()], [len(indvd.chrA)+len(indvd.chrB) for indvd in wfPop.values()] ## this calculate the fitness and load of an individual (i.e. two alleles). But the basis unit is an allele
            wL, loadL = [], []
            for indvd in wfPop.values():
                alleleA, alleleB = indvd.chrA, indvd.chrB
                wA, wB = wfPop.alleleFitness(alleleA), wfPop.alleleFitness(alleleB)
                loadA, loadB = len(alleleA), len(alleleB)
                wL.extend([wA, wB])
                loadL.extend([loadA, loadB])
            wmin, wmax, wmean = min(wL), max(wL), np.mean(wL)
            loadMin, loadMax, loadMean = min(loadL), max(loadL), np.mean(loadL)
            totNum, fixNum = len(site_count), Counter(site_count.values())[fixCount]
            wminL.append(wmin); wmaxL.append(wmax); wmeanL.append(wmean)
            loadMinL.append(loadMin); loadMaxL.append(loadMax); loadMeanL.append(loadMean)
            totNumL.append(totNum); fixNumL.append(fixNum)
        else: pass
        wfPop.WrightFisher_stepForward() ## 99.9% time
    recordL = list(zip(wminL, wmaxL, wmeanL, loadMinL, loadMaxL, loadMeanL, totNumL, fixNumL))
    return recordL
    
def wfSimToT_haploid_old(T, tStep, N, u, r, p, q, ld, sp, sq, ld_isConst, sp_isConst, sq_isConst, **kwargs):
    '''hr: homologous recombination rate, useless in this function'''
    ###### step 1: set up initial population
    kwargs_mutRandom = dict(r = r, p = p, q = q, ld = ld, sp = sp, sq = sq, ld_isConst = ld_isConst, sp_isConst = sp_isConst, sq_isConst = sq_isConst)
    randomMutFunc = partial(mutRandom, **kwargs_mutRandom)
    init_indvdL = [Individual_Haploid(u = u, w = 1, chrH = []) for i in range(N)]
    mutDict = dict()
    wfPop = Population_Haploid(indvdL = init_indvdL, mutDict = mutDict, initT = 0, u0 = u, randomMutFunc = randomMutFunc)
    ###### step 2: trace dynamics by time
    tL = []
    wminL, wmaxL, wmeanL = [], [], []
    loadMinL, loadMaxL, loadMeanL = [], [], []
    totNumL, fixNumL = [], []
    random.seed(); np.random.seed() ##### Because all the child processes are forked from the same main process and then np.random.seed (random.seed does not, but also add here) will produce the same values in different child processes. Need to add this command for parallel processes. 
    fixCount = N
    while wfPop.time <= T:
        if wfPop.time % tStep == 0: ## only record at some time points
            site_count = wfPop.sfs()
            cleanDict = {id: wfPop.mutDict[id] for id in site_count.keys()}; wfPop.mutDict = cleanDict
            # wfPop.clearMutDict()    ## clear out extinct mutations to save memory
            wL, loadL = [indvd.w for indvd in wfPop.values()], [len(indvd.chrH) for indvd in wfPop.values()]
            wmin, wmax, wmean = min(wL), max(wL), np.mean(wL)
            loadMin, loadMax, loadMean = min(loadL), max(loadL), np.mean(loadL)
            totNum, fixNum = len(site_count), Counter(site_count.values())[fixCount]
            wminL.append(wmin); wmaxL.append(wmax); wmeanL.append(wmean)
            loadMinL.append(loadMin); loadMaxL.append(loadMax); loadMeanL.append(loadMean)
            totNumL.append(totNum); fixNumL.append(fixNum)
        else: pass
        wfPop.WrightFisher_stepForward()  ## 99.9% time
    recordL = list(zip(wminL, wmaxL, wmeanL, loadMinL, loadMaxL, loadMeanL, totNumL, fixNumL))
    return recordL
    
def wfSimToT_haploid(T, tStep, N, u, r, p, q, ld, sp, sq, ld_isConst, sp_isConst, sq_isConst, **kwargs):
    '''hr: homologous recombination rate, useless in this function'''
    ###### step 1: set up initial population
    kwargs_mutRandom = dict(r = r, p = p, q = q, ld = ld, sp = sp, sq = sq, ld_isConst = ld_isConst, sp_isConst = sp_isConst, sq_isConst = sq_isConst)
    randomMutFunc = partial(mutRandom, **kwargs_mutRandom)
    init_indvdL = [Individual_Haploid(u = u, w = 1, chrH = []) for i in range(N)]
    mutDict = dict()
    wfPop = Population_Haploid(indvdL = init_indvdL, mutDict = mutDict, initT = 0, u0 = u, randomMutFunc = randomMutFunc)
    ###### step 2: trace dynamics by time
    tL = []
    wminL, wmaxL, wmeanL = [], [], []
    loadMinL, loadMaxL, loadMeanL = [], [], []
    totNumL, fixNumL = [], []
    random.seed(); np.random.seed() ##### Because all the child processes are forked from the same main process and then np.random.seed (random.seed does not, but also add here) will produce the same values in different child processes. Need to add this command for parallel processes. 
    fixCount = N
    while wfPop.time <= T:
        if wfPop.time % tStep == 0: ## only record at some time points
            wfPop.updateAll()
            fixW, fixNum, snpNum = wfPop.fixW, len(wfPop.fixMutDict), len(wfPop.mutDict)
            totNum = fixNum + snpNum
            wL, loadL = [fixW * indvd.w for indvd in wfPop.values()], [fixNum + len(indvd.chrH) for indvd in wfPop.values()]
            wmin, wmax, wmean = min(wL), max(wL), np.mean(wL)
            loadMin, loadMax, loadMean = min(loadL), max(loadL), np.mean(loadL)
            wminL.append(wmin); wmaxL.append(wmax); wmeanL.append(wmean)
            loadMinL.append(loadMin); loadMaxL.append(loadMax); loadMeanL.append(loadMean)
            totNumL.append(totNum); fixNumL.append(fixNum)
        else: pass
        wfPop.WrightFisher_stepForward()  ## 99.9% time
    recordL = list(zip(wminL, wmaxL, wmeanL, loadMinL, loadMaxL, loadMeanL, totNumL, fixNumL))
    return recordL
    
## performance and time analysis of a function
# T, tStep, N, u = 100, 20, 100, 0.03
# r, p, q, ld, sp, sq = 0.05, 0.03, 0.17, 1.1, 0.05, 0.01
# ld_isConst, sp_isConst, sq_isConst, hr = True, True, True, 0.5
# wfPop = Population_Haploid(indvdL = [Individual_Haploid(u = u, w = 1, chrH = []) for i in range(N)], mutDict = dict(), initT = 0, u0 = u, randomMutFunc = mutRandom)
# %load_ext line_profiler
# %lprun -f wfSimToT_haploid wfSimToT_haploid(T, tStep, N, u, r, p, q, ld, sp, sq, ld_isConst, sp_isConst, sq_isConst)
# %lprun -f wfPop.WrightFisher_stepForward wfPop.WrightFisher_stepForward()
# %lprun -f wfPop.indvdReproduce wfPop.indvdReproduce(wfPop[1])


def toTxt_lock(txtFile, lock, wfSimToT_func, wfSimToT_kwargs, repeatID):
    logging.debug('repeat %s is simulating' %repeatID)
    recordL = wfSimToT_func(**wfSimToT_kwargs)
    strText = '\t'.join(map(str, recordL)) + '\n'
    logging.debug('repeat %s simulation is done and writing out' %repeatID)
    with open(txtFile, 'a', buffering = 1) as f:
        with lock:
            f.write(strText)
            logging.debug('repeat %s is done' %repeatID)
def main_lock(repeat, procNum, wfSimToT_func, wfSimToT_kwargs):
    txtFile = 'wf{simType}Sim_N{N:.0e}_U{u:.4f}_r{r:.2f}_p{p:.2f}_q{q:.2f}_sp{sp:.4f}_sq{sq:.4f}_T{T:.0e}_const.txt'.format(**wfSimToT_kwargs)
    logName = txtFile.replace('.txt', '.log')
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(processName)-17s, PID-%(process)d, %(message)s', level=logging.DEBUG, filename = logName)
    logging.debug('Simulation results to %s' %txtFile)
    f = open(txtFile, 'w')
    f.write('## The second row is generation. Each tuple: (wmin, wmax, wmean, loadMin, loadMax, loadMean, totNum, fixNum)\n')
    T, tStep = wfSimToT_kwargs['T'], wfSimToT_kwargs['tStep']
    tList = np.arange(0, T+tStep, tStep); header = '\t'.join(map(str, tList)) + '\n'
    f.write(header); f.close()
    pools, lock = multiprocessing.Pool(processes = procNum), multiprocessing.Manager().Lock()
    for i in range(1, repeat + 1):
        pools.apply_async(toTxt_lock, (txtFile, lock, wfSimToT_func, wfSimToT_kwargs, i))
    pools.close(); pools.join()

if __name__ == '__main__':
    starT = time.time()
    repeat, procNum = args.rp, args.np
    ploidy = args.pld
    if ploidy == 1:
        wfSimToT_func = wfSimToT_haploid
        simType = 'Haploid'
    elif ploidy == 2:
        wfSimToT_func = wfSimToT_diploid
        simType = 'Diploid'
    else:
        raise ValueError('The ploidy (-pld) must be 1 or 2')
    wfSimToT_kwargs = dict(T = int(args.T), tStep = int(args.step), N = args.N, u = args.u, r = args.r, p = args.p, q = args.q, ld = args.ld, sp = args.sp, sq = args.sq, hr = args.hr, ld_isConst = args.ld_isConst, sp_isConst = args.sp_isConst, sq_isConst = args.sq_isConst, simType = simType)
    main_lock(repeat, procNum, wfSimToT_func, wfSimToT_kwargs)
    elapsedT = time.time() - starT
    logging.debug('Elapsed Time: %s' %elapsedT); print('Elapsed Time: %s' %elapsedT)
    
