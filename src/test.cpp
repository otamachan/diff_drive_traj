#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>

#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>

constexpr int kStateSize = 8;
constexpr int kConstraintSize = 5;
constexpr int kX = 0;
constexpr int kY = 1;
constexpr int kTH = 2;
constexpr int kV = 3;
constexpr int kW = 4;
constexpr int kAV = 5;
constexpr int kAW = 6;
constexpr int kDT = 7;

struct Configuration {
  int n{1};
  double max_v{0.3};
  double max_v_backward{0.3};
  double max_w{0.3};
  double max_av{0.1};
  double max_aw{0.1};
  double min_dt{0.05};
  double max_dt{0.15};
  double dt_ref{0.07};
  double dt_hysteresis{0.01};
};

struct State {
  Eigen::Vector2d start_vel;
  Eigen::Vector2d goal_vel;
  std::vector<Eigen::Vector3d> poses;
  std::vector<double> time_diffs;
};

class Variables : public ifopt::VariableSet {
 public:
  Variables(Configuration* config, State* init_state)
      : VariableSet(config->n * kStateSize, "x"),
        config_(config),
        init_state_(init_state) {
    x_.resize(config_->n * kStateSize, 0.0);
    for (int i = 0; i < config_->n; ++i) {
      // std::cerr << "x=" << init_state->poses.at(i).x()
      //           << ",y=" << init_state->poses.at(i).y()
      //           << ",th=" << init_state->poses.at(i).z();
      x_[i * kStateSize + kX] = init_state->poses.at(i).x();
      x_[i * kStateSize + kY] = init_state->poses.at(i).y();
      x_[i * kStateSize + kTH] = init_state->poses.at(i).z();
      if (i < config_->n - 1) {
        x_[i * kStateSize + kV] = (init_state->poses.at(i + 1).head<2>() -
                                   init_state->poses.at(i).head<2>())
                                      .norm() /
                                  init_state->time_diffs.at(i);
        x_[i * kStateSize + kDT] = init_state->time_diffs.at(i);
        // std::cerr << ",v=" << x_[i * kStateSize + kV];
        // std::cerr << ",dt=" << init_state->time_diffs.at(i);
      }
      // std::cerr << std::endl;
    }
  }

  void SetVariables(const VectorXd& x) override {
    VectorXd::Map(&x_[0], x_.size()) = x;
  };

  VectorXd GetValues() const override {
    VectorXd x = VectorXd::Map(&x_[0], x_.size());
    return x;
  };

  VecBound GetBounds() const override {
    VecBound bounds(GetRows());
    ifopt::Bounds av_bounds(-config_->max_av, config_->max_av);
    ifopt::Bounds aw_bounds(-config_->max_aw, config_->max_aw);
    ifopt::Bounds dt_bounds(config_->min_dt, config_->max_dt);
    // initial
    bounds.at(kX) = ifopt::Bounds(init_state_->poses.front().x(),
                                  init_state_->poses.front().x());
    bounds.at(kY) = ifopt::Bounds(init_state_->poses.front().y(),
                                  init_state_->poses.front().y());
    bounds.at(kTH) = ifopt::Bounds(init_state_->poses.front().z(),
                                   init_state_->poses.front().z());
    bounds.at(kV) =
        ifopt::Bounds(init_state_->start_vel.x(), init_state_->start_vel.x());
    bounds.at(kW) =
        ifopt::Bounds(init_state_->start_vel.y(), init_state_->start_vel.y());
    bounds.at(kAV) = av_bounds;
    bounds.at(kAW) = aw_bounds;
    bounds.at(kDT) = dt_bounds;
    // x
    int i = 1;
    for (; i < config_->n - 1; ++i) {
      bounds.at(i * kStateSize + kX) = ifopt::NoBound;   // x
      bounds.at(i * kStateSize + kY) = ifopt::NoBound;   // y
      bounds.at(i * kStateSize + kTH) = ifopt::NoBound;  // th
      bounds.at(i * kStateSize + kV) =
          ifopt::Bounds(-config_->max_v_backward, config_->max_v);  // v
      bounds.at(i * kStateSize + kW) =
          ifopt::Bounds(-config_->max_w, config_->max_w);  // w
      bounds.at(i * kStateSize + kAV) = av_bounds;
      bounds.at(i * kStateSize + kAW) = aw_bounds;
      bounds.at(i * kStateSize + kDT) = dt_bounds;
    }
    // final
    bounds.at(i * kStateSize + kX) = ifopt::Bounds(
        init_state_->poses.back().x(), init_state_->poses.back().x());
    bounds.at(i * kStateSize + kY) = ifopt::Bounds(
        init_state_->poses.back().y(), init_state_->poses.back().y());
    bounds.at(i * kStateSize + kTH) = ifopt::Bounds(
        init_state_->poses.back().z(), init_state_->poses.back().z());
    bounds.at(i * kStateSize + kV) =
        ifopt::Bounds(init_state_->goal_vel.x(), init_state_->goal_vel.x());
    bounds.at(i * kStateSize + kW) =
        ifopt::Bounds(init_state_->goal_vel.x(), init_state_->goal_vel.x());
    bounds.at(i * kStateSize + kAV) = ifopt::BoundZero;
    bounds.at(i * kStateSize + kAW) = ifopt::BoundZero;
    bounds.at(i * kStateSize + kDT) = ifopt::BoundZero;
    return bounds;
  }

 private:
  Configuration* config_;
  State* init_state_;
  std::vector<double> x_;
};

// Define Constraint
class Constraint : public ifopt::ConstraintSet {
 public:
  Constraint(Configuration* config)
      : ConstraintSet((config->n - 1) * kConstraintSize, "constraint"),
        config_(config) {}
  VectorXd GetValues() const override {
    std::vector<double> x(config_->n * kStateSize);
    VectorXd::Map(&x[0], x.size()) =
        GetVariables()->GetComponent("x")->GetValues();
    std::vector<double> gv = getAdFun().Forward(0, x);
    VectorXd g = VectorXd::Map(&gv[0], gv.size());
    return g;
  };
  VecBound GetBounds() const override {
    VecBound bounds(GetRows());
    for (int i = 0; i < (config_->n - 1) * kConstraintSize; ++i) {
      bounds.at(i) = ifopt::BoundZero;
    }
    return bounds;
  }
  void FillJacobianBlock(std::string var_set,
                         Jacobian& jac_block) const override {
    if (var_set == "x") {
      std::vector<double> x(config_->n * kStateSize);
      VectorXd::Map(&x[0], x.size()) =
          GetVariables()->GetComponent("x")->GetValues();
      std::vector<double> ad_jac = getAdFun().Jacobian(x);
      const int x_size = config_->n * kStateSize;
      auto copy_jac = [&jac_block, &ad_jac, x_size](int i, int j) mutable {
        // std::cerr << "(" << i << "," << j << ")=" << ad_jac[i * x_size + j]
        // << std::endl;
        jac_block.coeffRef(i, j) = ad_jac[i * x_size + j];
      };
      for (int i = 0; i < config_->n - 1; ++i) {
        // 0 = -x(n+1) + x(n) + v(n) * cos(th(n)) * dt(n)
        copy_jac(i * kConstraintSize + 0, (i + 1) * kStateSize + kX);  // x(i+1)
        copy_jac(i * kConstraintSize + 0, i * kStateSize + kX);        // x(i)
        copy_jac(i * kConstraintSize + 0, i * kStateSize + kTH);       // th(i)
        copy_jac(i * kConstraintSize + 0, i * kStateSize + kV);        // v(i)
        copy_jac(i * kConstraintSize + 0, i * kStateSize + kDT);       // dt(i)
        // 0 = -y(n+1) + y(n) + v(n) * sin(th(n)) * dt(n)
        copy_jac(i * kConstraintSize + 1, (i + 1) * kStateSize + kY);  // y(i+1)
        copy_jac(i * kConstraintSize + 1, i * kStateSize + kY);        // y(i)
        copy_jac(i * kConstraintSize + 1, i * kStateSize + kTH);       // th(i)
        copy_jac(i * kConstraintSize + 1, i * kStateSize + kV);        // v(i)
        copy_jac(i * kConstraintSize + 1, i * kStateSize + kDT);       // dt(i)
        // 0 = -th(n+1) + th(n) + w(n) * dt(n)
        copy_jac(i * kConstraintSize + 2,
                 (i + 1) * kStateSize + kTH);                     // th(i+1)
        copy_jac(i * kConstraintSize + 2, i * kStateSize + kTH);  // th(i)
        copy_jac(i * kConstraintSize + 2, i * kStateSize + kW);   // w(i)
        copy_jac(i * kConstraintSize + 2, i * kStateSize + kDT);  // dt(i)
        // 0 = -v(n+1) + v(n) + av(n) * dt(n)
        copy_jac(i * kConstraintSize + 3, (i + 1) * kStateSize + kV);  // v(i+1)
        copy_jac(i * kConstraintSize + 3, i * kStateSize + kV);        // v(i)
        copy_jac(i * kConstraintSize + 3, i * kStateSize + kAV);       // av(i)
        copy_jac(i * kConstraintSize + 3, i * kStateSize + kDT);       // dt(i)
        // 0 = -w(n+1) + w(n) + aw(n) * dt(n)
        copy_jac(i * kConstraintSize + 4, (i + 1) * kStateSize + kW);  // w(i+1)
        copy_jac(i * kConstraintSize + 4, i * kStateSize + kW);        // w(i)
        copy_jac(i * kConstraintSize + 4, i * kStateSize + kAW);       // aw(i)
        copy_jac(i * kConstraintSize + 4, i * kStateSize + kDT);       // dt(i)
      }
    }
  }

 private:
  CppAD::ADFun<double> getAdFun(void) const {
    std::vector<CppAD::AD<double>> ax(config_->n * kStateSize);
    CppAD::Independent(ax);
    std::vector<CppAD::AD<double>> ay((config_->n - 1) * kConstraintSize);
    for (int i = 0; i < config_->n - 1; ++i) {
      auto& xn = ax[(i + 1) * kStateSize + kX];
      auto& yn = ax[(i + 1) * kStateSize + kY];
      auto& thn = ax[(i + 1) * kStateSize + kTH];
      auto& vn = ax[(i + 1) * kStateSize + kV];
      auto& wn = ax[(i + 1) * kStateSize + kW];
      auto& x = ax[i * kStateSize + kX];
      auto& y = ax[i * kStateSize + kY];
      auto& th = ax[i * kStateSize + kTH];
      auto& v = ax[i * kStateSize + kV];
      auto& w = ax[i * kStateSize + kW];
      auto& av = ax[i * kStateSize + kAV];
      auto& aw = ax[i * kStateSize + kAW];
      auto& dt = ax[i * kStateSize + kDT];
      // 0 = -x(n+1) + x(n) + v(n) * cos(th(n)) * dt(n)
      ay[i * kConstraintSize + 0] = -xn + x + v * CppAD::cos(th) * dt;
      // 0 = -y(n+1) + y(n) + v(n) * sin(th(n)) * dt(n)
      ay[i * kConstraintSize + 1] = -yn + y + v * CppAD::sin(th) * dt;
      // 0 = -th(n+1) + th(n) + w(n) * dt(n)
      ay[i * kConstraintSize + 2] = -thn + th + w * dt;
      // 0 = -v(n+1) + v(n) + av(n) * dt(n)
      ay[i * kConstraintSize + 3] = -vn + v + av * dt;
      // 0 = -w(n+1) + w(n) + aw(n) * dt(n)
      ay[i * kConstraintSize + 4] = -wn + w + aw * dt;
    }
    CppAD::ADFun<double> f(ax, ay);
    return f;
  }
  Configuration* config_;
};

// Define Cost
class Cost : public ifopt::CostTerm {
 public:
  Cost(Configuration* config) : CostTerm("cost"), config_(config) {}

  double GetCost() const override {
    std::vector<double> x(config_->n * kStateSize);
    VectorXd::Map(&x[0], x.size()) =
        GetVariables()->GetComponent("x")->GetValues();
    std::vector<double> fv = getAdFun().Forward(0, x);
    return fv[0];
  };

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
    if (var_set == "x") {
      std::vector<double> x(config_->n * kStateSize);
      VectorXd::Map(&x[0], x.size()) =
          GetVariables()->GetComponent("x")->GetValues();
      std::vector<double> ad_jac = getAdFun().Jacobian(x);
      const int x_size = config_->n * kStateSize;
      auto copy_jac = [&jac, &ad_jac, x_size](int i, int j) mutable {
        jac.coeffRef(i, j) = ad_jac[i * x_size + j];
      };
      for (int i = 0; i < config_->n; ++i) {
        copy_jac(0, i * kStateSize + kDT);
      }
    }
  }

 private:
  CppAD::ADFun<double> getAdFun(void) const {
    std::vector<CppAD::AD<double>> ax(config_->n * kStateSize);
    CppAD::Independent(ax);
    std::vector<CppAD::AD<double>> ay(1);
    for (int i = 0; i < config_->n; ++i) {
      ay[0] += ax[i * kStateSize + kDT];
    }
    CppAD::ADFun<double> f(ax, ay);
    return f;
  }
  Configuration* config_;
};

inline double average_angle(double theta1, double theta2) {
  double x, y;
  x = cos(theta1) + cos(theta2);
  y = sin(theta1) + sin(theta2);
  if (x == 0 && y == 0)
    return 0;
  else
    return std::atan2(y, x);
}

void resize(const Configuration& config, State* state) {
  bool updated = true;
  while (updated) {
    updated = false;
    for (int i = 0; i < state->time_diffs.size(); ++i) {
      const double dt = state->time_diffs.at(i);
      if (dt > config.dt_ref > config.dt_hysteresis) {
        if (dt > 2 * config.dt_ref) {
          const double new_dt = dt / 2;
          const auto& p0 = state->poses.at(i);
          const auto& p1 = state->poses.at(i + 1);
          const Eigen::Vector2d new_t = (p0.head<2>() + p1.head<2>()) / 2;
          const double new_angle = average_angle(p0.z(), p1.z());
          state->time_diffs.at(i) = new_dt;
          state->poses.insert(state->poses.begin() + i + 1,
                              Eigen::Vector3d{new_t.x(), new_t.y(), new_angle});
          state->time_diffs.insert(state->time_diffs.begin() + i + 1, new_dt);
          --i;
          updated = true;
        } else {
          if (i < state->time_diffs.size() - 1) {
            state->time_diffs.at(i + 1) += dt - config.dt_ref;
          }
          state->time_diffs.at(i) = config.dt_ref;
        }
      } else if (dt < config.dt_ref - config.dt_hysteresis) {
        if (i < state->time_diffs.size() - 1) {
          state->time_diffs.at(i + 1) += dt;
          state->time_diffs.erase(state->time_diffs.begin() + i);
          state->poses.erase(state->poses.begin() + i + 1);
          i--;
        } else {
          state->time_diffs.at(i - 1) += dt;
          state->time_diffs.erase(state->time_diffs.begin() + i);
          state->poses.erase(state->poses.begin() + i);
        }
        updated = true;
      }
    }
  }
}

int main() {
  Configuration config;
  State state;

  state.poses.push_back(Eigen::Vector3d{0, 0, 0});
  state.start_vel << 0.0, 0;
  state.poses.push_back(Eigen::Vector3d{3, 3, -3.14});
  state.goal_vel << 0, 0;

  state.time_diffs.push_back(
      (state.poses.front().head<2>() - state.poses.back().head<2>()).norm() /
      config.max_v);

  resize(config, &state);
  config.n = state.poses.size();
  std::cerr << "n=" << config.n << std::endl;
  // Define problem
  ifopt::Problem nlp;
  nlp.AddVariableSet(std::make_shared<Variables>(&config, &state));
  nlp.AddConstraintSet(std::make_shared<Constraint>(&config));
  nlp.AddCostSet(std::make_shared<Cost>(&config));

  // Initialize solver
  ifopt::IpoptSolver ipopt;
  ipopt.SetOption("linear_solver", "mumps");
  ipopt.SetOption("jacobian_approximation", "exact");
  ipopt.SetOption("print_level", 5);  // supress log
  ipopt.SetOption("sb", "yes");       // supress startup message

  ipopt.Solve(nlp);
  auto v = nlp.GetOptVariables()->GetComponent("x")->GetValues();
  std::ofstream ofs_csv_file("traj.csv");
  for (int i = 0; i < config.n; ++i) {
    ofs_csv_file << v[i * kStateSize + kX] << "," << v[i * kStateSize + kY]
                 << "," << v[i * kStateSize + kTH] << ","
                 << v[i * kStateSize + kDT] << "," << v[i * kStateSize + kV]
                 << "," << v[i * kStateSize + kAV] << std::endl;
  }
  ofs_csv_file.close();
}
