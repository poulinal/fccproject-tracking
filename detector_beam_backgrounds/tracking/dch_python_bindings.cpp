#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include "DCH_info.h" // Replace with the actual path to the dd4hep module on your system
#include "/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/include/DDRec/DCH_info.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TVector3.h>
#include <TMath.h>

namespace py = pybind11;

// Helper function to convert TVector3 to Python tuple
py::tuple TVector3_to_tuple(const TVector3& vec) {
    return py::make_tuple(vec.X(), vec.Y(), vec.Z());
}

// Helper function to convert Python tuple/list to TVector3
TVector3 tuple_to_TVector3(const py::object& obj) {
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        if (py::len(obj) >= 3) {
            return TVector3(
                obj.cast<py::list>()[0].cast<double>(),
                obj.cast<py::list>()[1].cast<double>(),
                obj.cast<py::list>()[2].cast<double>()
            );
        }
    }
    throw std::runtime_error("Expected a tuple or list with 3 elements");
}

PYBIND11_MODULE(dch_module, m) {
    m.doc() = "Python bindings for DCH_info drift chamber geometry";
    

    // Expose the DCH_info_layer class
    py::class_<dd4hep::rec::DCH_info_struct::DCH_info_layer>(m, "DCH_info_layer")
        .def(py::init<>())
        .def_readwrite("layer", &dd4hep::rec::DCH_info_struct::DCH_info_layer::layer)
        .def_readwrite("nwires", &dd4hep::rec::DCH_info_struct::DCH_info_layer::nwires)
        .def_readwrite("height_z0", &dd4hep::rec::DCH_info_struct::DCH_info_layer::height_z0)
        .def_readwrite("width_z0", &dd4hep::rec::DCH_info_struct::DCH_info_layer::width_z0)
        .def_readwrite("radius_sw_z0", &dd4hep::rec::DCH_info_struct::DCH_info_layer::radius_sw_z0)
        .def_readwrite("radius_fdw_z0", &dd4hep::rec::DCH_info_struct::DCH_info_layer::radius_fdw_z0)
        .def_readwrite("radius_fuw_z0", &dd4hep::rec::DCH_info_struct::DCH_info_layer::radius_fuw_z0)
        .def("IsStereoPositive", &dd4hep::rec::DCH_info_struct::DCH_info_layer::IsStereoPositive)
        .def("StereoSign", &dd4hep::rec::DCH_info_struct::DCH_info_layer::StereoSign)
        .def("Pitch_z0", &dd4hep::rec::DCH_info_struct::DCH_info_layer::Pitch_z0);




    // Expose the DCH_info_struct class
    py::class_<dd4hep::rec::DCH_info_struct>(m, "DCH_info_struct")
        .def(py::init<>())

        //property binding
        .def_readwrite("Lhalf", &dd4hep::rec::DCH_info_struct::Lhalf)
        .def_readwrite("rin", &dd4hep::rec::DCH_info_struct::rin)
        .def_readwrite("rout", &dd4hep::rec::DCH_info_struct::rout)
        .def_readwrite("guard_inner_r_at_z0", &dd4hep::rec::DCH_info_struct::guard_inner_r_at_z0)
        .def_readwrite("guard_outer_r_at_zL2", &dd4hep::rec::DCH_info_struct::guard_outer_r_at_zL2)
        .def_readwrite("ncell0", &dd4hep::rec::DCH_info_struct::ncell0)
        .def_readwrite("ncell_increment", &dd4hep::rec::DCH_info_struct::ncell_increment)
        .def_readwrite("nlayersPerSuperlayer", &dd4hep::rec::DCH_info_struct::nlayersPerSuperlayer)
        .def_readwrite("nsuperlayers", &dd4hep::rec::DCH_info_struct::nsuperlayers)
        .def_readwrite("ncell_per_sector", &dd4hep::rec::DCH_info_struct::ncell_per_sector)
        .def_readwrite("twist_angle", &dd4hep::rec::DCH_info_struct::twist_angle)
        .def_readwrite("first_width", &dd4hep::rec::DCH_info_struct::first_width)
        .def_readwrite("first_sense_r", &dd4hep::rec::DCH_info_struct::first_sense_r)
        .def_readwrite("database", &dd4hep::rec::DCH_info_struct::database)

        // Setters
        .def("Set_lhalf", &dd4hep::rec::DCH_info_struct::Set_lhalf)
        .def("Set_rin", &dd4hep::rec::DCH_info_struct::Set_rin)
        .def("Set_rout", &dd4hep::rec::DCH_info_struct::Set_rout)
        .def("Set_guard_rin_at_z0", &dd4hep::rec::DCH_info_struct::Set_guard_rin_at_z0)
        .def("Set_guard_rout_at_zL2", &dd4hep::rec::DCH_info_struct::Set_guard_rout_at_zL2)
        .def("Set_ncell0", &dd4hep::rec::DCH_info_struct::Set_ncell0)
        .def("Set_ncell_increment", &dd4hep::rec::DCH_info_struct::Set_ncell_increment)
        .def("Set_nlayersPerSuperlayer", &dd4hep::rec::DCH_info_struct::Set_nlayersPerSuperlayer)
        .def("Set_nsuperlayers", &dd4hep::rec::DCH_info_struct::Set_nsuperlayers)
        .def("Set_ncell_per_sector", &dd4hep::rec::DCH_info_struct::Set_ncell_per_sector)
        .def("Set_twist_angle", &dd4hep::rec::DCH_info_struct::Set_twist_angle)
        .def("Set_first_width", &dd4hep::rec::DCH_info_struct::Set_first_width)
        .def("Set_first_sense_r", &dd4hep::rec::DCH_info_struct::Set_first_sense_r)
        
        // Getters and utilities
        .def("Get_ncells", &dd4hep::rec::DCH_info_struct::Get_ncells)
        .def("Get_phi_width", &dd4hep::rec::DCH_info_struct::Get_phi_width)
        .def("Get_cell_phi_angle", &dd4hep::rec::DCH_info_struct::Get_cell_phi_angle)
        .def("Get_nsuperlayer_minus_1", &dd4hep::rec::DCH_info_struct::Get_nsuperlayer_minus_1)
        .def("Radius_zLhalf", &dd4hep::rec::DCH_info_struct::Radius_zLhalf)
        .def("stereoangle_z0", &dd4hep::rec::DCH_info_struct::stereoangle_z0)
        .def("stereoangle_zLhalf", &dd4hep::rec::DCH_info_struct::stereoangle_zLhalf)
        .def("WireLength", &dd4hep::rec::DCH_info_struct::WireLength)
        
        // Main processing methods
        // .def("BuildLayerDatabase", &dd4hep::rec::DCH_info_struct::BuildLayerDatabase)
        .def("BuildLayerDatabase", [](dd4hep::rec::DCH_info_struct& self, bool verbose=true) {
            if (verbose) {
                // Print values before calling the original method
                std::cout << "\n=== DCH Parameters Before Building Layer Database ===\n";
                std::cout << "  Half length (Lhalf): " << self.Lhalf/dd4hep::mm << " mm\n";
                std::cout << "  Inner radius (rin): " << self.rin/dd4hep::mm << " mm\n";
                std::cout << "  Outer radius (rout): " << self.rout/dd4hep::mm << " mm\n";
                std::cout << "  Guard inner radius at z0: " << self.guard_inner_r_at_z0/dd4hep::mm << " mm\n";
                std::cout << "  Guard outer radius at zL2: " << self.guard_outer_r_at_zL2/dd4hep::mm << " mm\n";
                std::cout << "  Number of cells in first layer (ncell0): " << self.ncell0 << "\n";
                std::cout << "  Cell increment per superlayer: " << self.ncell_increment << "\n";
                std::cout << "  Number of cells per sector: " << self.ncell_per_sector << "\n";
                std::cout << "  Number of layers per superlayer: " << self.nlayersPerSuperlayer << "\n";
                std::cout << "  Number of superlayers: " << self.nsuperlayers << "\n";
                std::cout << "  Twist angle: " << self.twist_angle/dd4hep::deg << " deg\n";
                std::cout << "  First layer width: " << self.first_width/dd4hep::mm << " mm\n";
                std::cout << "  First sense radius: " << self.first_sense_r/dd4hep::mm << " mm\n";
                std::cout << "=================================================\n\n";
            }
            
            // Call the original BuildLayerDatabase method
            self.BuildLayerDatabase();
            
            // Optional: Print confirmation that database was built
            std::cout << "Layer database built successfully with " 
                      << self.database.size() << " layers.\n";
        }, py::arg("verbose") = true)
        // .def("IsValid", &dd4hep::rec::DCH_info_struct::IsValid)
        .def("IsDatabaseEmpty", &dd4hep::rec::DCH_info_struct::IsDatabaseEmpty)
        
        // //Vector calculation methods with Python-friendly wrappers
        // .def("Calculate_wire_vector_ez", [](const dd4hep::rec::DCH_info_struct& self, int ilayer, int nphi) {
        //     return TVector3_to_tuple(self.Calculate_wire_vector_ez(ilayer, nphi));
        // })
        // .def("Calculate_wire_z0_point", [](const dd4hep::rec::DCH_info_struct& self, int ilayer, int nphi) {
        //     return TVector3_to_tuple(self.Calculate_wire_z0_point(ilayer, nphi));
        // })
        // .def("Calculate_wire_phi_z0", &dd4hep::rec::DCH_info_struct::Calculate_wire_phi_z0)
        // .def("Calculate_hitpos_to_wire_vector", [](const dd4hep::rec::DCH_info_struct& self, int ilayer, int nphi, const py::object& hit_pos) {
        //     TVector3 hit_position = tuple_to_TVector3(hit_pos);
        //     return TVector3_to_tuple(self.Calculate_hitpos_to_wire_vector(ilayer, nphi, hit_position));
        // })

        // Expose the database

        // // Add a method to get a specific layer from the database
        // .def("get_layer", [](dd4hep::rec::DCH_info_struct& self, int layer_num) -> dd4hep::rec::DCH_info_struct::DCH_info_layer& {
        //     try {
        //         return self.database.at(layer_num);
        //     } catch (const std::out_of_range&) {
        //         throw py::key_error("Layer not found: " + std::to_string(layer_num));
        //     }
        // }, py::return_value_policy::reference)
        
        // // Add method to get all layer numbers
        // .def("get_layer_numbers", [](const dd4hep::rec::DCH_info_struct& self) {
        //     py::list result;
        //     for (const auto& pair : self.database) {
        //         result.append(pair.first);
        //     }
        //     return result;
        // })

        //get rin
        .def("get_rin", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.rin / dd4hep::mm;
        })
        //get rout
        .def("get_rout", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.rout / dd4hep::mm;
        })
        //get Lhalf
        .def("get_Lhalf", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.Lhalf / dd4hep::mm;
        })
        //get twist angle
        .def("get_twist_angle", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.twist_angle / dd4hep::deg;
        })
        //get nsuperlayers
        .def("get_nsuperlayers", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.nsuperlayers;
        })
        //get nlayersPerSuperlayer
        .def("get_nlayersPerSuperlayer", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.nlayersPerSuperlayer;
        })
        //get ncell0
        .def("get_ncell0", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.ncell0;
        })
        //get ncell_increment
        .def("get_ncell_increment", [](const dd4hep::rec::DCH_info_struct& self) {
            return self.ncell_increment;
        })

        // Add a method to get database as a list of dictionaries
        .def("get_database_as_list_dic", [](const dd4hep::rec::DCH_info_struct& self) {
            py::list result;
            
            for (const auto& [layer_num, layer_info] : self.database) {
                py::dict layer_dict;
                
                // Add layer number
                layer_dict["layer"] = layer_num;
                
                // Add all properties from DCH_info_layer
                layer_dict["nwires"] = layer_info.nwires;
                layer_dict["height_z0"] = layer_info.height_z0 /dd4hep::mm;
                layer_dict["width_z0"] = layer_info.width_z0 /dd4hep::mm;
                layer_dict["radius_sw_z0"] = layer_info.radius_sw_z0 /dd4hep::mm;
                layer_dict["radius_fdw_z0"] = layer_info.radius_fdw_z0 /dd4hep::mm;
                layer_dict["radius_fuw_z0"] = layer_info.radius_fuw_z0 /dd4hep::mm;
                
                // Add computed properties
                layer_dict["stereo_positive"] = layer_info.IsStereoPositive();
                layer_dict["stereo_sign"] = layer_info.StereoSign();
                
                // Add properties that take the radius as parameter
                layer_dict["pitch_z0"] = layer_info.Pitch_z0(layer_info.radius_sw_z0) /dd4hep::mm;
                
                // Add properties from the parent class that need this layer's info
                layer_dict["stereo_angle_z0"] = layer_info.StereoSign() * self.stereoangle_z0(layer_info.radius_sw_z0) / dd4hep::deg;
                layer_dict["radius_at_zLhalf"] = self.Radius_zLhalf(layer_info.radius_sw_z0) / dd4hep::mm;
                layer_dict["wire_length"] = self.WireLength(layer_num, layer_info.radius_sw_z0) / dd4hep::mm;
                
                result.append(layer_dict);
            }
            
            return result;
        });

    // Expose the DCH_info class (extension wrapper)
    py::class_<dd4hep::rec::DCH_info, dd4hep::rec::DCH_info_struct>(m, "DCH_info")
        .def(py::init<>());


        
    // Add needed constants for units
    m.attr("mm") = dd4hep::mm;
    m.attr("cm") = dd4hep::cm;
    m.attr("rad") = dd4hep::rad;
    m.attr("deg") = dd4hep::deg;
    m.attr("pi") = TMath::Pi();
}