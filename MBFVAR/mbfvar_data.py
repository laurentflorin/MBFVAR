import os
import sys

import pandas as pd
import numpy as np

from collections import deque
import copy
import itertools

    
class mbfvar_data:
    """
    Prepare multi-frequency time series data for MBFVAR estimation.

    This class handles the preprocessing and organization of time series data at
    different frequencies (e.g., quarterly, monthly, weekly) into a format suitable
    for the multi-frequency Bayesian VAR model.

    Parameters
    ----------
    data : list of pandas.DataFrame
        List of DataFrames, one for each frequency level, ordered from lowest to
        highest frequency. Each DataFrame should have:
        - Time series data with variables as columns
        - A datetime index or index that can be converted to datetime
        - Variables named consistently for identification

        Example: [quarterly_df, monthly_df, weekly_df]

    trans : list of numpy.ndarray
        List of transformation arrays, one for each frequency level. Each array
        specifies the transformation for each variable in that frequency:
        - 0: Take natural logarithm of the variable
        - 1: Divide the variable by 100 (for percentages/rates)

        Length of each array must match the number of variables in the corresponding
        DataFrame in `data`.

        Example: [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]

    frequencies : list of str
        List of frequency identifiers ordered from lowest to highest.
        Supported frequencies:
        - "Y": Yearly
        - "Q": Quarterly
        - "M": Monthly
        - "W": Weekly
        - "D": Daily

        The frequency ratios are automatically calculated based on standard
        calendar conventions (e.g., Q to M is 3, M to W is 4).

        Example: ["Q", "M", "W"]

    Attributes
    ----------
    YMX_list : collections.deque
        Original high-frequency data (all except lowest frequency)
    YM0_list : collections.deque
        Transformed high-frequency data as numpy arrays
    YQX_list : collections.deque
        Original low-frequency data (lowest frequency only)
    YQ0_list : collections.deque
        Transformed low-frequency data as numpy arrays
    frequencies : list of str
        The frequency identifiers passed during initialization
    freq_ratio_list : collections.deque
        Calculated frequency ratios between consecutive frequency levels
    varlist_list : collections.deque
        Variable names for each frequency combination
    YDATA_list : collections.deque
        Combined data matrices ready for model estimation

    Notes
    -----
    - The data transformations are applied automatically during initialization
    - For unsupported frequency combinations, you will be prompted to enter the
      frequency ratio manually
    - The class prepares the data for disaggregation from low to high frequency
      following the methodology in Schorfheide and Song (2015)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import MBFVAR
    >>>
    >>> # Load data at different frequencies
    >>> data_q = pd.read_excel("data.xlsx", sheet_name="Q", index_col=0)
    >>> data_m = pd.read_excel("data.xlsx", sheet_name="M", index_col=0)
    >>> data_w = pd.read_excel("data.xlsx", sheet_name="W", index_col=0)
    >>>
    >>> # Specify transformations (1 = divide by 100 for all variables)
    >>> trans = [
    ...     np.array([1, 1]),       # 2 quarterly variables
    ...     np.array([1, 1, 1]),    # 3 monthly variables
    ...     np.array([1, 1, 1])     # 3 weekly variables
    ... ]
    >>>
    >>> # Create data object
    >>> data_in = MBFVAR.mbfvar_data(
    ...     data=[data_q, data_m, data_w],
    ...     trans=trans,
    ...     frequencies=["Q", "M", "W"]
    ... )

    References
    ----------
    Schorfheide, F., & Song, D. (2015). Real-time forecasting with a mixed-frequency
    VAR. Journal of Business & Economic Statistics, 33(3), 366-380.
    """
    
    def __init__(self, data, trans, frequencies):
        
        
        
        # Creating lists of highfrequency data
        
        YMX_list = deque()
        YM0_list = deque()
        select_m_list = deque()
        vars_m_list = deque()
        YMh_list = deque()
        exc_list = deque()
        index_list = deque()
        
        
        for i in range(1, len(frequencies)):
            YMX_list.append(data[i])
            YM0_list.append(data[i].to_numpy(copy=True))
            select_m_list.append(trans[i])
            vars_m_list.append(data[i].columns[:])
            YMh_list.append(data[i].to_numpy())
            index_list.append(data[i].index)
        
        if not isinstance(index_list[-1], pd.DatetimeIndex):
            try:
                index_list[-1] = pd.to_datetime(index_list[-1])
            except:
                print("Index must be of the form 'YYYY-MM-DD'")
        
        
        input_data = copy.deepcopy(YMX_list)
        
        # Creating list of low frequency data
        # Here only the first entry with the lowest frequency data is generated
        # The other entries are generated during the estimation
        
        YQX_list = deque()
        YQ0_list = deque()
        select_q = deque()
        
        YQX_list.append(data[0])
        # copy=True ensures array is writable for in-place transformations
        YQ0_list.append(YQX_list[-1].to_numpy(copy=True))
        select_q.append(trans[0])
        vars_q = YQX_list[0].columns[:]
        
        input_data_Q =  copy.deepcopy(YQX_list[0])
        
        
        varlist_list = deque()
        select_list = deque()
        select_c_list =deque()
        
        Nm_list = deque()
        nv_list = deque()
        Nq_list = deque()
        
        Nq_list.append(YQX_list[0].shape[1])
        
        def safe_len(arr):
            arr = np.asarray(arr)
            if arr.shape == ():  # scalar
                return 0
            return len(arr)

        for i in range(len(YMX_list)-1):
            # Block i+1's low-frequency input is block i's COMPLETED panel,
            # i.e. all of block i's variables (its HF variables plus its own
            # accumulated LF set) -- the count and the transformation codes
            # must accumulate recursively along the chain. For two frequency
            # steps this coincides with the old "base quarterly + immediate
            # HF" formula, which is why the mis-sizing only ever surfaced
            # with three or more steps (B >= 3 blocks).
            Nq_list.append(Nq_list[i] + YMX_list[i].shape[1])
            if safe_len(select_m_list[i]) == 0:
                select_q.append( select_q[i])
            else:
                select_q.append(np.hstack((select_m_list[i], select_q[i])))
                
                
        for i in range(len(YMX_list)):
            if i > 0:
                new_list = [item for item in list(itertools.islice(select_m_list, 0, i+1)) if safe_len(item) > 0]
                select_list.append(np.hstack((np.hstack(list(reversed(new_list))), select_q[0])))
                rev_vars_m = list(itertools.islice(vars_m_list, 0, i+1))
                rev_vars_m.reverse()
                varlist_list.append(np.squeeze(np.hstack((np.hstack(rev_vars_m), vars_q)))) 
            else:
                if select_m_list[i].size:
                    select_list.append(np.hstack((select_m_list[i], select_q[0])))
                    varlist_list.append(np.hstack((vars_m_list[i], vars_q)))
                else:
                    select_list.append(select_q[0])
                    if len(vars_q.shape) > 1:
                        varlist_list.append(np.squeeze(vars_q))
                    else:
                        varlist_list.append(vars_q)
            Nm_list.append(int(np.shape(YM0_list[i])[1]))
            nv_list.append(int(Nm_list[i] + Nq_list[i]))
        
        select_list_sep = list(select_q)
        select_list_sep.extend(select_m_list)
        
        
        # Calculate the frequency ratios
        
        freq_ratio_list = deque()
        
        for freq in range(1,len(frequencies)):
            freq_lf = frequencies[freq-1]
            freq_hf = frequencies[freq]
            if freq_hf == "Q" and freq_lf == "Y":
                freq_ratio = 4
            elif freq_hf == "M" and freq_lf == "Y":
                freq_ratio = 12
            elif freq_hf == "W" and freq_lf == "Y":
                freq_ratio = 48
            elif freq_hf == "D" and freq_lf == "Y":
                freq_ratio = 260
            elif freq_hf == "M" and freq_lf == "Q":
                freq_ratio = 3
            elif freq_hf == "W" and freq_lf == "Q":
                freq_ratio = 12
            elif freq_hf == "W" and freq_lf == "M":
                freq_ratio = 4
            elif freq_hf == "D" and freq_lf == "W":
                freq_ratio = 5
                
            else:
                print("Higher frequency: ", freq_hf, " Lower Frequency: ", freq_lf, end ='\n')
                freq_ratio = input("Please enter frequency ratio")
            
            freq_ratio_list.append(freq_ratio)
        
        
        #performe data transformations
        
        for i in range(len(YM0_list)):
            if select_m_list[i].size:
                YM0_list[i][:,(select_m_list[i] == 1)] = YM0_list[i][:,(select_m_list[i] == 1)]/100
                YM0_list[i][:,(select_m_list[i] == 0)] = np.log(YM0_list[i][:,(select_m_list[i] == 0)])
        
        YQ0_list[0][:,(select_q[0] == 1)] = YQ0_list[0][:,(select_q[0] == 1)]/100
        YQ0_list[0][:,(select_q[0] == 0)] = np.log(YQ0_list[0][:,(select_q[0] == 0)])
        
        
        YM_list = copy.deepcopy(YM0_list)
        
        # Low frequency data in higher frequency
        YQ_list = deque()
        YQ_list.append(np.kron(YQ0_list[0], np.ones((freq_ratio_list[0],1))))
        
        Tstar_list = deque()
        T_list = deque()
        YDATA_list = deque()
        
        if YM_list[0].size:
            Tstar_list.append(YM_list[0].shape[0])
            YDATA_list.append(np.full((Tstar_list[0],nv_list[0]), np.nan))
            YDATA_list[0][:,:Nm_list[0]] = YM_list[0]
        else:
            Tstar_list.append(YQ_list[0].shape[0])
            YDATA_list.append(np.full((0,nv_list[0]), np.nan))
            
        T_list.append(YQ_list[0].shape[0])
        #for freq in range(1,len(self.frequencies)-1):
        #    T_list.append(np.kron(YMX_list[freq-1], np.ones((freq_ratio_list[freq],1))).shape[0])
        
        
        if YDATA_list[0].size:     
            YDATA_list[0][:T_list[0],Nm_list[0]:] = YQ_list[0]   
        else:
            YDATA_list[0] = YQ_list[0] 
            
        
        #attach variables to instance
        
        self.YMX_list = YMX_list
        self.YM0_list = YM0_list
        self.select_m_list = select_m_list
        self.vars_m_list = vars_m_list
        self.YMh_list = YMh_list
        self.exc_list = exc_list
        self.index_list = index_list
        self.frequencies = frequencies
        self.YQX_list = YQX_list
        self.YQ0_list = YQ0_list
        self.select_q = select_q
        self.input_data_Q =  input_data_Q
        self.varlist_list = varlist_list
        self.select_list = select_list
        self.select_c_list = select_c_list
        self.Nm_list = Nm_list
        self.nv_list = nv_list
        self.Nq_list = Nq_list
        self.select_list_sep = select_list_sep
        self.freq_ratio_list = freq_ratio_list
        self.YQ_list = YQ_list
        self.Tstar_list = Tstar_list
        self.T_list = T_list
        self.YDATA_list = YDATA_list
        self.YM_list = YM_list
        self.input_data = input_data
        self.trans = trans