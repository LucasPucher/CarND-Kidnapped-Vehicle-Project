/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * 
 * Adapted on: June 12, 2019
 * Adapted by: Lucas Pucher
 * 
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "map.h"
#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	/**
	* TODO: Set the number of particles. Initialize all particles to 
	*   first position (based on estimates of x, y, theta and their uncertainties
	*   from GPS) and all weights to 1. 
	* TODO: Add random Gaussian noise to each particle.
	* NOTE: Consult particle_filter.h for more information about this method 
	*   (and others in this file).
	*/
	num_particles = 100;  // TODO: Set the number of particles

	// Create a normal (Gaussian) distribution for x, y and theta
	std::default_random_engine gen;
	
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);


	Particle p;

	for (int i = 0; i < num_particles; ++i) {
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		particles.push_back(p);
		weights.push_back(0.0);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
	/**
	* TODO: Add measurements to each particle and add random Gaussian noise.
	* NOTE: When adding noise you may find std::normal_distribution 
	*   and std::default_random_engine useful.
	*  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	*  http://www.cplusplus.com/reference/random/default_random_engine/
	*/

	/* Prepare noise calculation */
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);		

	/* Do some calculations first */
	double theta = yaw_rate * delta_t;
	double rc = 0.0;
	
	if(fabs(yaw_rate) > 0.001) {
		rc = velocity / yaw_rate;
	}
	
	for (int i = 0; i < num_particles; i++) {
		Particle *p = &particles[i];
		
		if(fabs(yaw_rate) > 0.001) {
			p->x = p->x + rc * (sin(p->theta + theta) - sin(p->theta));
			p->y = p->y + rc * (cos(p->theta) - cos(p->theta + theta));
			p->theta = p->theta + theta;
		}
		else {
			p->x += velocity * delta_t * cos(p->theta);
			p->y += velocity * delta_t * sin(p->theta);
			// Theta will stay the same due to no yaw_rate
		}
		p->x += dist_x(gen);
		p->y += dist_y(gen);
		p->theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

	for (unsigned int i = 0; i < observations.size(); i++) {

		LandmarkObs *o = &observations[i];
		o->id = -1;

		/* initalize best distance to a big number */
		double best_dist = 10000.0;

		for (unsigned int j = 0; j < predicted.size(); j++) {
			
			/* get distance between observation and map landmark */
			double new_dist = dist(o->x, o->y, predicted[j].x, predicted[j].y);

			/* find the predicted landmark nearest the current observed landmark */
			if (new_dist < best_dist) {
				best_dist = new_dist;
				o->id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	
	unsigned int lm_size = map_landmarks.landmark_list.size();
	
	
	for (int i = 0; i < num_particles; i++) {

		Particle *p = &particles[i];
		
		/* First clear all associated predictions */
		p->sense_x.clear();
		p->sense_y.clear();
		p->associations.clear();

		/* Filter landmarks by range */
		vector<LandmarkObs> filtered_lm;

		/* Iterate through all landmarks and only add landmarks within range */
		for (unsigned int j = 0; j < lm_size; j++) {
			// get id and x,y coordinates
			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;

			double dist_lm = dist(p->x, p->y, lm_x, lm_y);
			if (dist_lm < sensor_range) {
				filtered_lm.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
			}
		}

		/* Create a list of observation for the function dataAssociation. Observations are transformed to map coordinates */
		vector<LandmarkObs> m_observations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double t_x = cos(p->theta)*observations[j].x - sin(p->theta)*observations[j].y + p->x;
			double t_y = sin(p->theta)*observations[j].x + cos(p->theta)*observations[j].y + p->y;
			m_observations.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		/* Find a landmark for each observation - observations without landmarks have an index -1*/
		dataAssociation(filtered_lm, m_observations);

		/* Weight must be set to 1.0 because we will multiply it */
		particles[i].weight = 1.0;

		/* Iterate through all measurements */
		for (unsigned int j = 0; j < m_observations.size(); j++) {
			
			double x_obs, y_obs, x_pred, y_pred;
			
			x_obs = m_observations[j].x;
			y_obs = m_observations[j].y;

			int associated_pred = m_observations[j].id;

			/* get the x,y coordinates of the prediction associated with the current observation */
			for (unsigned int k = 0; k < filtered_lm.size(); k++) {
				if (filtered_lm[k].id == associated_pred) {
					x_pred = filtered_lm[k].x;
					y_pred = filtered_lm[k].y;
					
					/* Fill the associated predictions */
					p->sense_x.push_back(x_pred);
					p->sense_y.push_back(y_pred);
					p->associations.push_back(filtered_lm[k].id);
				}	
			}

			/* calculate weight  */
			double obs_w = multiv_prob(std_landmark[0], std_landmark[1], x_obs, y_obs, x_pred, y_pred); 

			/* the total weight is the product of individual weights*/
			particles[i].weight *= obs_w;
		}
		weights[i] = particles[i].weight;
	}
	
	/* Normalize weights */
	double weight_sum = 0.0;
	for (int i = 0; i < num_particles; i++) {
		weight_sum += weights[i];
	}
	for (int i = 0; i < num_particles; i++) {
		weights[i] = weights[i]/weight_sum;
	}
	
	
}

void ParticleFilter::resample() {
	/**
	* TODO: Resample particles with replacement with probability proportional 
	*   to their weight. 
	* NOTE: You may find std::discrete_distribution helpful here.
	*   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	*/

	vector<Particle> new_particles (num_particles);

	/* Use discrete distribution to return particles by weight */
	std::random_device rd;
	std::default_random_engine gen(rd());
	for (int i = 0; i < num_particles; ++i) {
	std::discrete_distribution<int> index(weights.begin(), weights.end());
	new_particles[i] = particles[index(gen)];

	}

	/* Replace old particles with the resampled particles */
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double w;
  w = gauss_norm * exp(-exponent);
    
  return w;
}