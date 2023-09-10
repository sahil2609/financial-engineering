#include <bits/stdc++.h>
using namespace std;

double f(double x, double y){
	return (y*(1 - exp(x)))/(1 + exp(x));
}


double fourth_order_rk(double h = 0.001, double x0 = 0, double y0 =3.0, double xt = 1.0){
	int n = xt/h + 1;
	vector<double>x(n), y(n);
	x[0] = x0, y[0] =y0;
	int i=1;
	while(i<n){
		x[i] = x[i-1] + h;
		double K1 =  f(x[i-1], y[i-1]);
		double K2 = f(x[i-1] + 0.50*h, y[i-1] + 0.5*K1*h);
		double K3 = f(x[i-1] + h*0.50,  y[i-1] + 0.5*K2*h);
		double K4 = f(x[i-1] + h, y[i-1] + K3*h);
		y[i] = y[i-1] + (h/6.0)*(K1 + 2*K2 + 2*K3 + K4);
		i++;
	}
	return y[n-1];
}


int main(){
	cout << "Required value of x(1) taking h = 0.001 : ";
	printf("%2.5f\n", fourth_order_rk());
	cout << "Required value of x(-1) taking h = -0.001 : ";
	printf("%2.5f\n", fourth_order_rk(-0.001,0,3.0, -1.0));

	cout << "\nNOTE: The values are rounded up to 5 decimal places\n";
}
