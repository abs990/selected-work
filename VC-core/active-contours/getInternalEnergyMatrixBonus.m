function [Ainv] = getInternalEnergyMatrixBonus(nPoints, alpha, beta, gamma)

%compute coefficients
i_diff_2_entry = beta;
i_diff_1_entry = -1 * alpha - 4 * beta;
i_entry = 2 * alpha + 6 * beta;

%allocate space for A and define identity matrix
A = zeros(nPoints,nPoints);
identityN = eye(nPoints); 

%define fill positions
num_row_fill_positions = 5;
row_fill_positions = [nPoints-1 nPoints 1 2 3];
fill_values = [i_diff_2_entry i_diff_1_entry i_entry i_diff_1_entry i_diff_2_entry];

%fill A
for i = 1:nPoints
    for j = 1:num_row_fill_positions
        A(i,row_fill_positions(j)) = fill_values(j);
        if(row_fill_positions(j) == nPoints)
            row_fill_positions(j) = 1;
        else
            row_fill_positions(j) = row_fill_positions(j) + 1;
        end    
    end    
end

%final result
M = A + gamma * identityN;
Ainv = inv(M);
