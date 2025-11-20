#ifndef SDU_ESTIMATORS_DISTURBANCE_OBSERVERS_MOMENTUM_OBSERVER_HPP
#define SDU_ESTIMATORS_DISTURBANCE_OBSERVERS_MOMENTUM_OBSERVER_HPP

#include <Eigen/Core>
#include <functional>
#include <memory>  // for std::shared_ptr

namespace sdu_estimators::disturbance_observers
{

  class MomentumObserver
  {
   public:
    /**
     * @brief Create the momentum observer from a model.
     * It is assumed that the model provides the following methods:
     * - get_inertia_matrix(const Eigen::VectorXd& q) -> Eigen::MatrixXd
     * - get_coriolis(const Eigen::VectorXd& q, const Eigen::VectorXd& qd) -> Eigen::MatrixXd
     * - get_gravity(const Eigen::VectorXd& q) -> Eigen::VectorXd
     * - get_friction(const Eigen::VectorXd& qd) -> Eigen::VectorXd
     *
     * Which should return the torque contributions for the given state.
     *
     * @tparam ModelType Type of the dynamic model
     * @param model Shared pointer to the dynamic model
     * @param dt Time step for the observer
     * @param K Gain matrix for the observer
     * @throws std::invalid_argument if the size of K does not match the model DOF.
     */
    template<typename ModelType>
    MomentumObserver(const std::shared_ptr<ModelType>& model, double dt, const Eigen::VectorXd& K)
        : MomentumObserver(
              [model](const Eigen::VectorXd& q) { return model->get_inertia_matrix(q); },
              [model](const Eigen::VectorXd& q, const Eigen::VectorXd& qd) { return model->get_coriolis(q, qd); },
              [model](const Eigen::VectorXd& q) { return model->get_gravity(q); },
              [model](const Eigen::VectorXd& qd) { return model->get_friction(qd); },
              dt,
              K)
    {
    }
    
    /**
     * @brief Create the momentum observer from function handles.
     * The functions should return the torque contributions for the given state.
     * 
     * @param get_inertia_matrix [in] Function to get the inertia matrix B(q)
     * @param get_coriolis [in] Function to get the Coriolis forces C(q, qd)
     * @param get_gravity [in] Function to get the gravity forces g(q)
     * @param get_friction [in] Function to get the friction forces f(qd)
     * @param dt [in] Time step for the observer
     * @param K [in] Gain matrix for the observer
     */
    MomentumObserver(
        const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& get_inertia_matrix,
        const std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& get_coriolis,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& get_gravity,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& get_friction,
        double dt,
        const Eigen::VectorXd& K)
        : get_inertia_matrix(get_inertia_matrix),
          get_coriolis(get_coriolis),
          get_gravity(get_gravity),
          get_friction(get_friction),
          _dt(dt),
          _K(K.asDiagonal()),
          _initialized(false)
    {
      if (K.size() != get_inertia_matrix(Eigen::VectorXd::Zero(K.size())).rows())
      {
        throw std::invalid_argument("Gain matrix K size must match the model DOF.");
      }
      _r = Eigen::VectorXd::Zero(K.size());
      _internal_r_sum = Eigen::VectorXd::Zero(K.size());
      _last_q = Eigen::VectorXd::Zero(K.size());
    }

    /**
     * @brief Reset the observer internal state.
     */
    void reset()
    {
      _r.setZero();
      _internal_r_sum.setZero();
      _initialized = false;
      _last_q.setZero();
    }

    /**
     * @brief Update the observer with new measurements.
     * 
     * @param q [in] Joint positions
     * @param qd [in] Joint velocities
     * @param tau [in] Measured joint torques
     */
    void update(const Eigen::VectorXd& q, const Eigen::VectorXd& qd, const Eigen::VectorXd& tau)
    {
      Eigen::MatrixXd B = get_inertia_matrix(q), C = get_coriolis(q, qd);
      Eigen::VectorXd g = get_gravity(q);
      Eigen::VectorXd friction = get_friction(qd);

      Eigen::VectorXd beta = g - C.transpose() * qd + friction;
      Eigen::VectorXd Feq = C * qd + g;  // Compute the equivalent force
      Eigen::VectorXd momentum = B * qd;

      if (!_initialized)
      {
        _initialized = true;
        _internal_r_sum = momentum;
      }
      _internal_r_sum += _dt * (tau - beta + _r);
      _r = _K * (momentum - _internal_r_sum);
      _last_q = q;
    }

    /**
     * @brief Update the observer with new measurements.
     * 
     * @param q [in] Joint positions
     * @param qd [in] Joint velocities
     * @param tau_m [in] Measured joint torques
     */
    void update(const std::vector<double>& q, const std::vector<double>& qd, const std::vector<double>& tau_m)
    {
      const Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
      const Eigen::VectorXd qd_eigen = Eigen::Map<const Eigen::VectorXd>(qd.data(), qd.size());
      const Eigen::VectorXd tau_eigen = Eigen::Map<const Eigen::VectorXd>(tau_m.data(), tau_m.size());
      update(q_eigen, qd_eigen, tau_eigen);
    }

    /**
     * @brief Get the estimated external joint torques.
     * @return Estimated external joint torques
     */
    Eigen::VectorXd estimatedTorques() const
    {
      return _r;  // Return the external joint torque estimate
    }

    /**
     * @brief Get the estimated acceleration.
     * 
     * @param q [in] Joint positions
     * @param qd [in] Joint velocities
     * @param tau [in] Measured joint torques
     * @return Eigen::VectorXd 
     */
    Eigen::VectorXd getAccEstimate(const Eigen::VectorXd& q, const Eigen::VectorXd& qd, const Eigen::VectorXd& tau) const
    {
      Eigen::MatrixXd B = get_inertia_matrix(q), C = get_coriolis(q, qd);
      
      Eigen::VectorXd g = get_gravity(q);
      Eigen::VectorXd friction = get_friction(qd);

      Eigen::VectorXd ddq = B.inverse() * (tau - C * qd - g - friction + estimatedTorques());
      return ddq;
    }

    /**
     * @brief Zero the external force/torque estimate.
     * This will adjust the internal state to make the current external torque estimate zero.
     */
    void zeroExternalFT()
    {
      _internal_r_sum = _K.inverse() * _r + _internal_r_sum;
    }

   private:
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> get_inertia_matrix;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> get_coriolis;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_gravity;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_friction;

    double _dt;
    Eigen::MatrixXd _K;  // Gain matrix for the observer
    Eigen::VectorXd _r;
    Eigen::VectorXd _internal_r_sum;  // Internal state for the observer
    Eigen::VectorXd _last_q;
    bool _initialized;
  };
}  // namespace sdu_estimators::disturbance_observers

#endif