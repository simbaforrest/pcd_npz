#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <climits>
#include <map>

#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "cnpy.h"

using namespace std;

void help(int argc, char **argv)
{
    cout << argv[0] << " <input.<pcd|npz>|input_folder> [output.<npz|pcd>|output_folder] [xyz=xyz] [normal=normal_est]" << endl;
}

template<typename PointType>
int pcd2npz(const pcl::PointCloud<PointType>& pcd, const string& fname, const map<string,string>& names=map<string,string>());

template<typename PointType>
int npz2pcd(const cnpy::npz_t& npz, pcl::PointCloud<PointType>& pcd, const map<string,string>& names=map<string,string>());

#define FILL_BUF(nch, pcd, buf, field) \
    { \
        int di=0; \
        for(int ci=0; ci<nch; ++ci) { \
            for(int hi=0; hi<h; ++hi) { \
                for(int wi=0; wi<w; ++wi, ++di) { \
                    float tmp = pcd.at(wi, hi).field[ci]; \
                    if (pcl_isnan(tmp)) tmp=0.f; \
                    buf[di] = tmp;\
                } \
            } \
        } \
    }

#define FILL_PCD(nch, buf, pcd, field) \
    { \
        int di=0; \
        for(int ci=0; ci<nch; ++ci) { \
            for(int hi=0; hi<h; ++hi) { \
                for(int wi=0; wi<w; ++wi, ++di) { \
                    float tmp = buf[di]; \
                    if (tmp==0) tmp=std::numeric_limits<float>::quiet_NaN(); \
                    pcd.at(wi, hi).field[ci] = tmp;\
                } \
            } \
        } \
    }

inline string find_name(const string& target, const string& default_name, const map<string, string>& names)
{
    if (names.find(target) != names.end()) {
        return names.find(target)->second;
    }
    return default_name;
}

template<>
int pcd2npz(const pcl::PointCloud<pcl::PointXYZ>& pcd, const string& fname, const map<string,string>& names)
{
    try {
        const int h = pcd.height;
        const int w = pcd.width;
        const int npts = h*w;

        vector<float> xyz(3 * npts);
        float *xyz_buf = &(xyz[0]);
        FILL_BUF(3, pcd, xyz_buf, data);

        const unsigned int shape[]={3, h, w};
        string xyz_name = find_name("xyz", "data", names);
        cnpy::npz_save(fname.c_str(), xyz_name.c_str(), xyz_buf, shape, 3, "w");
    } catch (...) {
        return -1;
    }
    return 0;
}

template<>
int pcd2npz(const pcl::PointCloud<pcl::PointNormal>& pcd, const string& fname, const map<string,string>& names)
{
    try {
        const int h = pcd.height;
        const int w = pcd.width;
        const int npts = h*w;

        vector<float> xyz(3 * npts);
        float *xyz_buf = &(xyz[0]);
        FILL_BUF(3, pcd, xyz_buf, data);
        vector<float> normal(3 * npts);
        float *normal_buf = &normal[0];
        FILL_BUF(3, pcd, normal_buf, normal);

        const unsigned int shape[]={3, h, w};
        const string xyz_name = find_name("xyz", "data", names);
        const string normal_name = find_name("normal", "normal", names);
        cnpy::npz_save(fname.c_str(), xyz_name.c_str(), xyz_buf, shape, 3, "w");
        cnpy::npz_save(fname.c_str(), normal_name.c_str(), normal_buf, shape, 3, "a");
    } catch (...) {
        return -1;
    }
    return 0;
}

template<>
int npz2pcd(const cnpy::npz_t& npz, pcl::PointCloud<pcl::PointXYZ>& pcd, const map<string,string>& names)
{
    try {
        const string xyz_name = find_name("xyz", "data", names);
        if(npz.find(xyz_name)==npz.end()) {
            cerr << "xyz field (field name: "<<xyz_name
                 <<") not find in loaded npz file!"<<endl;
            return -1;
        }
        const cnpy::NpyArray& xyz_arr = npz.find(xyz_name)->second;
        assert(xyz_arr.shape.size()==3 && xyz_arr.shape[0]==3);
        const int h = xyz_arr.shape[1];
        const int w = xyz_arr.shape[2];
        pcd.resize(h*w);
        pcd.height= h;
        pcd.width = w;
        float* xyz_buf = reinterpret_cast<float*>(xyz_arr.data);
        FILL_PCD(3, xyz_buf, pcd, data);
    } catch(...) {
        return -1;
    }
    return 0;
}

template<>
int npz2pcd(const cnpy::npz_t& npz, pcl::PointCloud<pcl::PointNormal>& pcd, const map<string,string>& names)
{
    try {
        const string xyz_name = find_name("xyz", "data", names);
        if(npz.find(xyz_name)==npz.end()) {
            cerr << "xyz field (field name: "<<xyz_name
                 <<") not find in loaded npz file!"<<endl;
            return -1;
        }
        const cnpy::NpyArray& xyz_arr = npz.find(xyz_name)->second;
        assert(xyz_arr.shape.size()==3 && xyz_arr.shape[0]==3);
        const int h = xyz_arr.shape[1];
        const int w = xyz_arr.shape[2];
        pcd.resize(h*w);
        pcd.height= h;
        pcd.width = w;
        float* xyz_buf = reinterpret_cast<float*>(xyz_arr.data);
        FILL_PCD(3, xyz_buf, pcd, data);
        
        const string normal_name = find_name("normal", "normal", names);
        if(npz.find(normal_name)==npz.end()) {
            cerr << "normal field (field name: "<<normal_name
                 <<") not find in loaded npz file!"<<endl;
            return -1;
        }
        const cnpy::NpyArray& normal_arr = npz.find(normal_name)->second;
        assert(normal_arr.shape.size()==3 && normal_arr.shape[0]==3 
            && normal_arr.shape[1]==h && normal_arr.shape[2]==w);
        float* normal_buf = reinterpret_cast<float*>(normal_arr.data);
        FILL_PCD(3, normal_buf, pcd, normal);
    } catch(...) {
        return -1;
    }
    return 0;
}

int process(string fnameIn, string fnameOut, map<string, string> names)
{
    int ret=0;
    if (fnameIn.find(".npz")!=string::npos) { //npz2pcd
        if (fnameOut.size() == 0) {
            fnameOut = fnameIn;
            fnameOut.replace(fnameIn.size()-4, 4, ".pcd");
        }
        cnpy::npz_t npz = cnpy::npz_load(fnameIn.c_str());
        if(npz.find(names["normal"])==npz.end()) {
            pcl::PointCloud<pcl::PointXYZ> pcd;
            ret = npz2pcd(npz, pcd, names);
            pcl::io::savePCDFileBinary(fnameOut, pcd);
        } else {
            pcl::PointCloud<pcl::PointNormal> pcd;
            ret = npz2pcd(npz, pcd, names);
            pcl::io::savePCDFileBinary(fnameOut, pcd);
        }
        npz.destruct();
        if (ret==0) {
            cout << "converted: " << fnameIn << " -> " << fnameOut << endl;
        } else {
            cout << "failed to convert: " << fnameIn << endl;
        }
    } else if (fnameIn.find(".pcd")!=string::npos) { //pcd2npz
        pcl::PCLPointCloud2 msg;
        pcl::io::loadPCDFile(fnameIn.c_str(), msg);
        bool has_normal = false;
        for(int i=0; i<msg.fields.size(); ++i) {
            if(msg.fields[i].name.find("normal")!=string::npos) {
                has_normal = true;
                break;
            }
        }
        if(fnameOut.size()==0) {
            fnameOut=fnameIn;
            fnameOut.replace(fnameIn.size() - 4, 4, ".npz");
        }
        if(has_normal) {
            pcl::PointCloud<pcl::PointNormal> pcd;
            pcl::fromPCLPointCloud2(msg, pcd);
            ret = pcd2npz(pcd, fnameOut, names);
        } else {
            pcl::PointCloud<pcl::PointXYZ> pcd;
            pcl::fromPCLPointCloud2(msg, pcd);
            ret = pcd2npz(pcd, fnameOut, names);
        }
        if(ret==0) {
            cout << "converted: " << fnameIn << " -> " << fnameOut << endl;
        } else {
            cout << "failed to convert: " << fnameIn << endl;
        }
    } else {
        cout << "not implemented for this file type!" << endl;
        return -1;
    }

    return ret;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        help(argc, argv);
        return -1;
    }

    string fnameIn(argv[1]);
    string fnameOut;
    if (argc > 2) fnameOut = string(argv[2]);

    map<string, string> names;
    names["xyz"] = "data";
    names["normal"] = "normal";

    for (int i = 3; i < argc; ++i) {
        string argvi(argv[i]);
        size_t pos = argvi.find("=");
        if (pos != string::npos) {
            string key = argvi.substr(0, pos);
            string val = argvi.substr(pos + 1);
            names[key] = val;
        }
    }

    return process(fnameIn, fnameOut, names);
}