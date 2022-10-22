#ifndef TIMERS_HPP
#define TIMERS_HPP

#include <chrono>
#include <iostream>

class Timer {
	public:
		std::string id;
		std::string unit;
		std::chrono::time_point<std::chrono::system_clock> t_start;
		std::chrono::time_point<std::chrono::system_clock> t_stop;
	
		Timer(char *id, char *unit) {
			this->id = std::string(id);
			this->unit = std::string(unit);
			this->t_start = std::chrono::high_resolution_clock::now();
		}

		void reset(char *id) {
			this->id = std::string(id);
			this->t_start = std::chrono::high_resolution_clock::now();
		}

		void stop() {
			t_stop = std::chrono::high_resolution_clock::now();
			auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(t_stop - t_start).count();
			if (this->unit == "mus") {
			   std::cout << "Timer " << this->id << ": " << delta / 1e3 << " mus" << std::endl;
			} else if (this->unit == "ms") {
			   std::cout << "Timer " << this->id << ": " << delta / 1e6 << " ms" << std::endl;
			} else if (this->unit == "s") {
			   std::cout << "Timer " << this->id << ": " << delta / 1e9 << " s" << std::endl;
			} else {
			   std::cout << "Timer " << this->id << ": " << delta << " ns" << std::endl;
			}
		}
};

#endif
