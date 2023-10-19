#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <optional>
#include <vector>
#include <cmath>
#include <unordered_map>

class TimeProfiler
{
public:
  typedef std::shared_ptr<TimeProfiler> Ptr;
  typedef std::shared_ptr<const TimeProfiler> ConstPtr;
  typedef std::string Key;

  typedef struct Timer
  {
    // Use std::optional so we can tell if the start and stop values
    // were set properly. If they are not, then we return -1
    std::optional<std::chrono::high_resolution_clock::time_point> start;
    std::optional<std::chrono::high_resolution_clock::time_point> stop;

    // To compute statistics about timing (e.g., mean)
    int n_calls = 0;
    double total_duration = 0.0;
    std::vector<double> durations;

    Timer()
    {
    }

    double duration()
    {
      if (stop && start) // checks if variables were set via tic/toc
      {
        durations.push_back(
            (std::chrono::duration<double>(*stop - *start)).count());
        return durations.back();
      }
      else
        return -1;
    }

    double sum()
    {
      double sum = 0.0;
      for (int i = 0; i < durations.size(); ++i)
      {
        sum += durations[i];
      }
      return sum;
    }

    double mean()
    {
      return sum() / durations.size();
    }

    double stddev()
    {
      double stddev = 0.0;
      for (int i = 0; i < durations.size(); ++i)
      {
        stddev += std::pow(durations[i] - mean(), 2);
      }

      if (nCalls() <= 1)
        return 0;

      return sqrt(stddev / (nCalls() - 1));
    }

    double nCalls()
    {
      return durations.size();
    }

    void printTimingStatistics(const Key &k)
    {
      std::cout << "Statistics for " << k << ":" << std::endl;
      std::cout << "\t"
                << "Total time spent: \t" << sum() << std::endl;
      std::cout << "\t"
                << "Number of calls:\t" << nCalls() << std::endl;
      std::cout << "\t"
                << "Avg. time:\t\t" << mean() << std::endl;
      std::cout << "\t"
                << "Standard deviation:\t" << stddev() << std::endl;
    }

  } timer_t;
  typedef timer_t Value;

  TimeProfiler()
  {
    reset();
  }

  ~TimeProfiler()
  {
    if (!save_)
      return;

    // Open file for writing
    std::ofstream ofs(path_);
    ofs << "Key,Sum,Calls,Mean,StdDev\n";
    for (auto const &pair : timers_)
    {
      Key k = pair.first;
      Value v = pair.second;
      ofs << k << "," << std::to_string(v.sum()) << ","
          << std::to_string(v.nCalls()) << "," << std::to_string(v.mean())
          << "," << std::to_string(v.stddev()) << "\n";
    }
    ofs.close();
  }

  void tic(const Key &key)
  {
    if (!exists(key))
    {
      addTimer(key);
    }

    Value timer = timers_.at(key);
    timers_[key].start =
        std::optional<std::chrono::high_resolution_clock::time_point>(
            std::chrono::high_resolution_clock::now());
  }

  double toc(const Key &key)
  {
    if (!exists(key))
    {
      return -1;
    }

    timers_[key].stop =
        std::optional<std::chrono::high_resolution_clock::time_point>(
            std::chrono::high_resolution_clock::now());
    return timers_[key].duration();
  }

  bool save(const std::string &directory, const std::string &filename)
  {
    path_ = std::filesystem::path{directory};
    if (!std::filesystem::is_directory(path_))
    {
      std::filesystem::create_directory(path_);
    }

    path_ /= filename;
    save_ = true;

    return save_;
  }

private:
  void reset()
  {
    timers_.clear();
  }

  void addTimer(const Key &key)
  {
    Value v;
    timers_.insert(std::make_pair(key, v));
  }

  bool exists(const Key &key) const
  {
    return !(timers_.find(key) == timers_.end());
  }

  std::unordered_map<Key, Value> timers_;
  bool save_ = false;
  std::filesystem::path path_;
};
