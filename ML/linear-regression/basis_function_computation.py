import numpy as np

"""
    Matrix will have have each row with basis values corresponding to each data point, upto required degree.
    1. Compute required dimensions of matrix
    2. Fill first column with 1s
    3. Thereafter, fill monomials in increasing order of powers
"""
def getBasisFunctionValueMatrix(x, max_degree):
    #Compute dimensions
    N_points, N_features = np.shape(x)
    N_columns = max_degree * N_features + 1

    #Allocate space for basis values
    basis_value_x = np.zeros(shape=(N_points,N_columns))
    
    #Fill 1st column with 1s
    basis_value_x[:,0] = 1

    #compute values for each row
    for row in np.arange(N_points):
        col_start = 1
        #fill values for all monomials upto required degree for current data point
        for pow in np.arange(1,max_degree+1):
            col_end = col_start + N_features
            #fill values for current power
            basis_value_x[row,col_start:col_end] = np.power(x[row],pow)    
            #update for next iteration
            col_start = col_end

    #return final matrix
    return basis_value_x

"""
    Function to get polynomial matrix for degree 2 - including mix terms as well
"""
def getSecondDegreePolynomialMatrix(x):
    #allocate space for matrix
    N_points, N_features = np.shape(x)
    N_columns = int(0.5*(N_features+1)*(N_features+2))
    basis_value_x = np.zeros(shape=(N_points,N_columns))

    #fill first column with 1's
    basis_value_x[:,0] = 1

    #fill entries for first degree
    degree_1_lim = N_features + 1
    basis_value_x[:,1:degree_1_lim] = x

    #fill second degree terms for each data point
    for row in np.arange(0,N_points):
        column = degree_1_lim
        for f_index in np.arange(0,N_features):
            #compute number of mixed terms to be filled
            N_mixed = N_features - f_index - 1
            #fill monomial
            f_value = np.float64(x[row,f_index])
            basis_value_x[row,column] = np.square(f_value)
            column = column + 1
            #fill mixed terms if needed
            if N_mixed > 0:
                basis_value_x[row,column:(column+N_mixed)] = f_value * x[row, (f_index+1):]
                column = column + N_mixed

    #return matrix
    return basis_value_x        
