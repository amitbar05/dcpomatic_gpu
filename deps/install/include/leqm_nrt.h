/*
    leqm-nrt is a  non-real-time implementation
    of Leq(M) measurement according to ISO 21727:2004(E)
    "Cinematography -- Method of measurement of perceived
    loudness of motion-picture audio material"

    Copyright (C) 2011-2013, 2017-2018 Luca Trisciani

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <mutex>
#include <cmath>
#include <memory>

namespace leqm_nrt {

class Sum
{
public:
	void sum_samples(std::vector<double> const& input_samples, std::vector<double> const& c_input_samples, int nsamples)
	{
		_mutex.lock();
		_nsamples += nsamples;
		for (auto i = 0; i < nsamples; i++) {
			_sum  += input_samples[i];
			_csum += c_input_samples[i];
		}
		_mutex.unlock();
	}

	int nsamples() const
	{
		return _nsamples;
	}

	/*
	   How the final offset is calculated without reference to a test tone:
	   P0 is the SPL reference 20 uPa
	   Reference SPL is RMS ! So 85 SPL over 20 uPa is 10^4.25 x 0.000020 = 0.355655882 Pa (RMS),
	   but Peak value is 0.355655882 x sqr(2) = 0.502973372 that is 20 x log ( 0.502973372 / 0.000020) = 88.010299957
	   To that one has to add the 20 dB offset of the reference -20dBFS: 88.010299957 + 20.00 = 108.010299957

	   But ISO 21727:2004(E) ask for a reference level "measured using an average responding meter". So reference level is not 0.707, but 0.637 = 2/pi
	*/

	double rms() const
	{
		return 20 * log10(mean()) + 108.010299957;
	}

	double leqm() const
	{
		return 20 * log10(cmean()) + 108.010299957;
	}

private:
	double mean() const
	{
		return pow(_sum / _nsamples, 0.500);
	}

	double cmean() const
	{
		return pow(_csum / _nsamples, 0.500);
	}

	double _csum = 0.0; // convolved sum
	double _sum = 0.0; // flat sum
	int _nsamples = 0;
	std::mutex _mutex;
};


struct Result
{
	Result(int status_)
		: status(status_)
	{}

	Result(double leq_m_, double leq_nw_)
		: status(0)
		, leq_m(leq_m_)
		, leq_nw(leq_nw_)
	{}

	/** 0 on success, or
	 *
	 * -100: Either channel_corrections contained a different number of
	 *  calibrations than; number of channels in the file or it was empty and  the
	 *  program cannot infer one from the number of channels. Please specify a
	 *  values in channel_corrections.
	 *
	 * -101: Failed to open the sound file.
	 *
	 * -102: buffer_size_ms is not an integer number of samples at the sound file's rate.
	 */
	int status;

	double leq_m;
	double leq_nw;
};


#ifdef LEQM_NRT_WITH_LIBSNDFILE
Result calculate_file(
	std::string sound_filename,
	std::vector<double> channel_corrections,
	int buffer_size_ms,
	int number_of_filter_interpolation_points,
	int num_cpu
	);
#endif


double convert_log_to_linear_single(double in);


class BadBufferSizeError : public std::runtime_error
{
public:
	BadBufferSizeError()
		: std::runtime_error("Buffer size does not correspond to an integer number of samples")
	{}
};


class BadChannelCorrectionsError : public std::runtime_error
{
public:
	BadChannelCorrectionsError()
		: std::runtime_error("Incorrect number of channel corrections given, and no defaults are available")
	{}
};


class Worker;


class Calculator
{
public:
	Calculator(
		int channels,
		int sample_rate,
		int bits_per_sample,
		std::vector<double> channel_corrections,
		int buffer_size_ms,
		int number_of_filter_interpolation_points,
		int num_cpu
	);

	~Calculator()
	{
		_workers.clear();
	}

	Calculator(Calculator&) = delete;
	Calculator(Calculator&&) = delete;
	bool operator=(Calculator&) = delete;
	bool operator=(Calculator&&) = delete;

	void add(std::vector<double> samples);

	double leq_m();
	double leq_nw();

private:
	void process_buffer();

	int _channels;
	std::vector<double> _channel_corrections;
	int _num_cpu;
	std::vector<std::shared_ptr<Worker>> _workers;
	Sum _sum;
	std::vector<double> _ir;
	std::vector<double> _buffer;
	size_t _buffer_free_offset = 0;
};

}

