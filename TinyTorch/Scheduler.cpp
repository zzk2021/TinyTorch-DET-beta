/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Scheduler.h"

namespace TinyTorch::optim::lr_scheduler {

float StepLR::getLr() {
  if (lastEpoch_ == 0 || lastEpoch_ % stepSize_ != 0) {
    return optimizer_.getLr();
  }
  return optimizer_.getLr() * gamma_;
}

}  // namespace TinyTorch::optim::lr_scheduler
