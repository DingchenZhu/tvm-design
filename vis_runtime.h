#ifndef VIS_RUNTIME_
#define VIS_RUNTIME_

#include <tvm/runtime/module.h>
namespace tvm {
namespace runtime {
namespace contrib {

class VISModule : public ModuleNode {
 public:
  /*! \brief The dla runtime module. */
  VISModule(std::string symbol, Array<String> consts)
      : symbol_name_(std::move(symbol)),
        const_names_(std::move(consts)) {}

  ~VISModule() {}
  const char* type_key() const override { return "vis_vpu"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {return PackedFunc(nullptr);}



 private:
  bool initialized_{false};
  /*! \brief The only subgraph name for this module. */
  const std::string symbol_name_;
  /*! \brief The required constant names. */
  const Array<String> const_names_;
};


}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif