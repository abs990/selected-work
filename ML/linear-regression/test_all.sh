chmod 775 *
sh environment_check.sh
for file in polynomial_regression.py polynomial_regression_1d.py polynomial_regression_reg.py
do
    echo "Running "$file
    ./$file
done