import pystan
import pickle

#Run the "Eight schools example from Gelman 2013 et al. which studies coaching effects from eight schools

#Use different branches depending on whether the model should be compiled and saved or loaded
is_write_to_file = True
is_load_from_file = False

if is_write_to_file:
	schools_code = """
	data {
		int<lower=0> J; //number of schools
		vector[J] y; // estimated treatment effects
		vector<lower=0>[J] sigma; // s.e. of effect estimates
	}
	parameters {
		real mu;
		real<lower=0> tau;
		vector[J] eta;
	}
	transformed parameters {
		vector[J] theta;
		theta = mu + tau*eta;
	}
	model {
		eta ~normal(0,1);
		y ~ normal(theta,sigma);
	}
	"""

	schools_dat = {'J': 8,
					'y': [28,8,-3,7,-1,1,18,12],
					'sigma': [15,10,16,11,9,11,10,18]}
					
	sm = pystan.StanModel(model_code = schools_code)
	fit = sm.sampling(data=schools_dat, iter = 1000,chains = 4)

	#Save the model to file
	model_filename = 'stanex1_model.pkl'
	with open(model_filename,'wb') as f:
		pickle.dump(sm, f)

if is_load_from_file:
	#Try loading the file
	model_filename = 'stanex1_model.pkl'
	sm = pickle.load(open(model_filename,'rb'))

	new_data = dict(J = 6, y = [28,8,-3,7,-1,1],sigma = [15,10,16,11,9,11])
	fit2 = sm.sampling(data=new_data)
