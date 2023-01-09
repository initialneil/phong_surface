#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "triangle_walk.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/* Triwalk to walk on triangle mesh
*  - pass numpy: https://www.jianshu.com/p/c912a0a59af9
*/
class Triwalk
{
public:
    Triwalk(Eigen::MatrixXi& F) 
    {
        if (F.rows() == 0) {
            throw std::runtime_error("[ERROR] F is empty");
        }
        if (F.cols() != 3) {
            throw std::runtime_error(std::string("[ERROR] F.cols() must be 3 != ") + std::to_string(F.cols()));
        }

        printf("[Triwalk] init mesh with F(%d, %d)\n", F.rows(), F.cols());
        m_triwalk.initTriangleMesh(F);

    }

    ~Triwalk() {}

private:
    prometheus::TriangleWalk m_triwalk;
};


PYBIND11_MODULE(triwalk, m) {
    m.doc() = R"pbdoc(
        Walking on Triangle Mesh
        -----------------------
        - reference: https://arxiv.org/abs/2007.04940
        - init
        - update
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // verbose
    m.def("add", [](int i, int j) { return i + j; });


    py::class_<Triwalk>(m, "Triwalk")
    .def(py::init<Eigen::MatrixXi&>());
    // .def("__call__", &Triwalk::operator());

}
