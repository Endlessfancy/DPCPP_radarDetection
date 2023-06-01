//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
// #include "exception_handler.hpp"
// #include "kernel.hpp"
// #include "Complex.hpp"

using namespace sycl;
using namespace std;


// constexpr float kTol = 0.001;
// static const int num_elements = 10000;

#define SampleSize 100              // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128               // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 1                 // the frame number
#define RxSize 4                    // the rx size, which is usually 4
#define cspd 3.0e8                     // the speed of light 
#define pi 3.141592653589793               // pi 



double F0=77e9;                      // the initial frequency
double mu=5.987e12;                  // FM slope
double chirp_sample=100;             // the sample number in a chirp, suggesting it should be the power of 2
double Fs=2.0e6;                     // sampling frequency
double num_chirp=128;                // the chirp number in a frame, suggesting it should be the the power of 2
double Framenum=90;                  // the frame number 
double Tr=64e-6;                     // the interval of the chirp
double fr=1/ Tr;                     // chirp repeating frequency,
double lamda=cspd/F0;                   // lamda of the initial frequency 
double d=0.5*lamda;                  // rx_wire array distance. When it is equal to the half of the wavelength, the 
                                     // maximum unambiguous Angle can reach -90° to +90°
static const int Tx_num = 1;
static const int Rx_num = 4;                      // the rx size, which is usually 4
 

// class CustomDeviceSelector {
//  public:
//   CustomDeviceSelector(std::string vendorName) : vendorName_(vendorName){};
//   int operator()(const device &dev) {
//     int device_rating = 0;
//     //We are querying for the custom device specific to a Vendor and if it is a GPU device we
//     //are giving the highest rating as 3 . The second preference is given to any GPU device and the third preference is given to
//     //CPU device. 
//     if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) !=
//                         std::string::npos))
//       device_rating = 3;
//     else if (dev.is_gpu())
//       device_rating = 2;
//     else if (dev.is_cpu())
//       device_rating = 1;
//     return device_rating;
//   };

//  private:
//   std::string vendorName_;
// };


// int readandreshape(char *filepath, Complex *Data_reshape);
// void ReshapeComplex(short *OriginalArray, Complex *Reshape, int size);
// void writeComplexBin(Complex *buf, int size);
// void writeBin(char *path, short *buf, int size);
// void readBin(char *path, short *buf, int size);
// int getBinSize(char *path);
class Complex {
 private:

 public:
  double real, imag;

  Complex() {
    real = 0;
    imag = 0;
  }

  Complex(int x, int y) {
    real = x;
    imag = y;
  }

  // Overloading the  != operator
  friend bool operator!=(const Complex& a, const Complex& b) {
    return (a.real != b.real) || (a.imag != b.imag);
  }

  // The function performs Complex number multiplication and returns a Complex
  // object.
  Complex complex_mul(const Complex& obj) const {
    return Complex(((real * obj.real) - (imag * obj.imag)),
                    ((real * obj.imag) + (imag * obj.real)));
  }

  // Overloading the ostream operator to print the objects of the Complex
  // object
  friend ostream& operator<<(ostream& out, const Complex& obj) {
    out << "(" << obj.real << " : " << obj.imag << "i)";
    return out;
  }
};

  Complex GetComplex(double r, double i){
    Complex temp;
    temp.real = r;
    temp.imag = i;
    return temp;  
  }

  Complex Complex_SUB(Complex a, Complex b){
    Complex temp;
    temp.real = a.real - b.real;
    temp.imag = a.imag - b.imag;
    return temp;
  }

  Complex Complex_MUL(Complex a, Complex b){
    Complex temp;
    temp.real = a.real * b.real - a.imag * b.imag;
    temp.imag = a.real * b.imag + a.imag * b.real;
    return temp;
  }

  double Complex_mol(Complex *a){
    return sqrt(a->real*a->real+a->imag*a->imag);
  }

int FindAbsMax(Complex *ptr, int size){
    int maxidx = 0;
    double maxval = 0;
    double absval;
    for (int i = 0; i < size; i++){
        absval = Complex_mol((ptr+i));
        if(absval > maxval){
            maxval = absval;
            maxidx = i;
        }
    }
    return maxidx;
}


// reshape the input bin file
// Input: *OriginalArray: the input of short bin file, size: the real size in form of short
// Output: *Reshape: reshape the input in form of complex
void ReshapeComplex(short *OriginalArray, Complex *Reshape, int size){
    int i, j, k;
    //int cnt = 0;
    Complex *buf_complex = (Complex *)malloc(size*sizeof(Complex)/2);
    short *ptr;
    Complex *complex_ptr = buf_complex;
    // reshape into 2 form of complex

    // #

    // # pragma omp for nowait
    // #pragma omp for schedule(static, 64) nowait
    for(i= 0 ;i < size; i+=4){
        ptr = OriginalArray+ i;
        complex_ptr->real = (double)*(ptr);  
        complex_ptr->imag  = (double)*(ptr +2 ); 
        complex_ptr++;  
        complex_ptr->real = (double)*(ptr +1 );  
        complex_ptr->imag  = (double)*(ptr +3 );   
        complex_ptr++;    
    }      
    Complex *Reshape_ptr;
    // change the sequence of the array, reshape it in form of Rx instead of frame
    
    // # pragma omp for nowait
    #pragma omp for schedule(static, 64) nowait
    for (i = 0; i <RxSize; i++){
        for (j = 0; j <FrameSize*ChirpSize; j++){
            for(k =0; k< SampleSize; k++){
                Reshape_ptr = (Reshape+i*FrameSize*ChirpSize*SampleSize+j*SampleSize+k); 
                complex_ptr = (buf_complex+j*RxSize*SampleSize+i*SampleSize+k);
                Reshape_ptr->real = complex_ptr->real;
                Reshape_ptr->imag = complex_ptr->imag;
            }
        }
    }
    free(buf_complex);
    return;
}

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

int main() {
  // try {
    char filepath[] = "./fhy_s.bin";
    FILE *infile;
    if ((infile = fopen(filepath, "rb")) == NULL){
        printf("\nCan not open the path: %s \n", filepath);
    }
    auto readsize = ChirpSize*SampleSize*RxSize*2 ; 
    auto size = ChirpSize*SampleSize*RxSize;
    auto extendSize = 128*128;

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(default_selector_v, exception_handler, property::queue::in_order());
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    // run the kernel 
    //# initialize data on host
    //# Explicit USM allocation using malloc_device
    // implicit USM
    // short *DataHost_Frm0_read = static_cast<short *>(malloc(size * sizeof(short)));
    // short *DataHost_Frm_reda = static_cast<short *>(malloc(size * sizeof(short)));
    short *DataShared_Frm0_read = malloc_shared<short>(size, q);
    short *DataShared_Frm_read = malloc_shared<short>(size, q);
    Complex *DataShared_Frm0_reshape = malloc_shared<Complex>(size, q);
    Complex *DataShared_Frm_reshape = malloc_shared<Complex>(size, q);
    // Complex *DataShared_Frm = malloc_shared<Complex>(size, q);
    Complex *DataShared_Rx = malloc_shared<Complex>(ChirpSize*SampleSize, q);
    Complex *DataShared_Rx_extended = malloc_shared<Complex>(extendSize, q);
    double *Dis = malloc_shared<double>(89, q);
    
    // Complex *DataShared_FFT = malloc_shared<Complex>(ChirpSize*SampleSize, q);

    // Complex *DataHost_Frm0 = static_cast<Complex *>(malloc(size * sizeof(Complex)));
    // Complex *DataHost_Frm = static_cast<Complex *>(malloc(size * sizeof(Complex)));
    // Complex *DataDevc_Frm0 = malloc_device<Complex>(size, q);
    // Complex *DataDevc_Frm = malloc_device<Complex>(size, q);

    // get the data
    fread(DataShared_Frm0_read, sizeof(short), readsize, infile);
    ReshapeComplex(DataShared_Frm0_read, DataShared_Frm0_reshape, readsize);

    // q.memcpy(DataDevc_Frm0, DataDecv_Frm0, sizeof(Complex) * size).wait();

    for (int frm = 0; frm < 89; frm++){
        //   q.memcpy(DataDevc_Frm, DataDecv_Frm,, sizeof(Complex) * size).wait();

        // read the data 
        if(fread(DataShared_Frm_read, sizeof(short), ChirpSize*SampleSize*RxSize*2, infile) == 0){
            break;
        }
        // reshape 
        ReshapeComplex(DataShared_Frm_read, DataShared_Frm_reshape, readsize);
        auto e = q.submit([&](handler& h) {
            // Preprocess: minus the frm0 data
            h.parallel_for(range<1>(SampleSize * ChirpSize), [=](id<1> i) { 
                DataShared_Rx[i] = Complex_SUB(DataShared_Frm_reshape[i], DataShared_Frm0_reshape[i]);
            });
            h.parallel_for(range<1>(extendSize), [=](id<1> i) { 
                if(i < SampleSize * ChirpSize ){
                    DataShared_Rx_extended[i] = DataShared_Rx[i];
                }
                else{
                    DataShared_Rx_extended[i] = GetComplex(0, 0);
                }
            });
            // Distance detection
            // Do FFT for the complex data:
            // h.parallel_for(range<1>(SampleSize * ChirpSize), [=](id<1> i) {  
            //     ReshapeComplex(DataDevc_Frm0, DataDevc_Frm0, 2*size);
            //     ReshapeComplex(DataDevc_Frm, DataDevc_Frm, 2*size);
            // }).wait();
            // Find max of the data 
            // h.parallel_for(nd_range<1>(SampleSize * ChirpSize), [=](nd_item<1> item) {
            //     auto idx = item.get_local_id(0);
            //     int maxDisIdx;
            //     FindAbsMax(item, DataShared_FFT, maxDisIdx, idx);
            //     item.barrier(access::fence_space::local_space);
            // });
            
        });
        
        int maxidx = FindAbsMax(DataShared_Rx_extended, floor(0.4*extendSize));
        // int extendsize = getextendsize(ChirpSize*SampleSize);
        double Fs_extend = Fs * extendSize/(ChirpSize*SampleSize);
        // int maxidx = FindAbsMax(Data_fft, floor(0.4*extendSize));
        int maxDisidx = maxidx*(ChirpSize*SampleSize)/extendSize;
        double maxDis = cspd*(((double)maxDisidx/extendSize)*Fs_extend)/(2*mu);
        Dis[frm] = maxDis;
    
    }
    
    //# print output
    for (int i = 0; i < 89; i++) std::cout << Dis[i] << "\n";

    // free for the USM
    free(DataShared_Frm0_read, q);
    free(DataShared_Frm_read, q);
    free(DataShared_Frm0_reshape, q);
    free(DataShared_Frm_reshape, q);
    free(DataShared_Rx, q);
    free(DataShared_Rx_extended, q);
    free(Dis, q);
    

  // } catch (exception const &e) {
  //   std::cout << "An exception is caught while adding two vectors.\n";
  //   std::terminate();
  // }
  // } catch (exception const &e) {
  //   // Catches exceptions in the host code
  //   std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

  //   // Most likely the runtime couldn't find FPGA hardware!
  //   if (e.code().value() == CL_DEVICE_NOT_FOUND) {
  //     std::cerr << "If you are targeting an FPGA, please ensure that your "
  //                  "system has a correctly configured FPGA board.\n";
  //     std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
  //     std::cerr << "If you are targeting the FPGA emulator, compile with "
  //                  "-DFPGA_EMULATOR.\n";
  //   }
  //   std::terminate();
  // }

  // Summarize results
  // if (correct == kArraySize) {
  //   std::cout << "PASSED: results are correct\n";
  // } else {
  //   std::cout << "FAILED: results are incorrect\n";
  // }

  std::cout << "Test successfully completed on device.\n";
  return 0;
}


// /// C read the binfile size
// int getBinSize(char *path){
//     int  size = 0;
//     FILE  *fp = fopen(path, "rb");
//     if (fp){
//         fseek(fp, 0, SEEK_END);
//         size = ftell(fp);
//         fclose(fp);
//     }
//     return size;
// }

// // C read the bin data in size of short
// void readBin(char *path, short *buf, int size){
//     FILE *infile;
//     if ((infile = fopen(path, "rb")) == NULL){
//         printf("\nCan not open the path: %s \n", path);
//     }
//     fread(buf, sizeof(short), size, infile);
//     fclose(infile);
// }

// // C write the bin data in size of short
// void writeBin(char *path, short *buf, int size){
//     FILE *outfile;
//     if ((outfile = fopen(path, "wb")) == NULL){
//         printf("\nCan not open the path: %s \n", path);
//     }
//     fwrite(buf, sizeof(short), size, outfile);
//     fclose(outfile);
// }

// // C write the bin data in size of Complex
// void writeComplexBin(Complex *buf, int size){
//     char saveFilePath_real[] = "./res_reshape_real.bin"  ;
//     char saveFilePath_imag[] = "./res_reshape_imag.bin" ;

//     short *real = (short*)malloc(size*sizeof(short));
//     short *imag = (short*)malloc(size*sizeof(short));

//     Complex *ptr= buf;
//     printf("\nwrite size = %d\n ",size);
//     for (int i=0; i< size; i++){
//         *(real+i) =(short)(ptr->real);
//         *(imag+i) =(short)(ptr->imag);
//         ptr++;
//     }

//     writeBin(saveFilePath_imag, imag, size);
//     int size_imag = getBinSize(saveFilePath_imag);
//     printf("write2 finished, size = %d short\n", size_imag);

//     writeBin(saveFilePath_real, real, size);
//     int size_real = getBinSize(saveFilePath_real);
//     printf("write1 finished, size = %d short\n", size_real);
// }

// // reshape the input bin file
// // Input: *OriginalArray: the input of short bin file, size: the real size in form of short
// // Output: *Reshape: reshape the input in form of complex
// void ReshapeComplex(short *OriginalArray, Complex *Reshape, int size){
//     int i, j, k, l;
//     int cnt = 0;
//     Complex *buf_complex = (Complex *)malloc(size*sizeof(Complex)/2);
//     short *ptr;
//     Complex *complex_ptr = buf_complex;
//     // reshape into 2 form of complex

//     // #

//     // # pragma omp for nowait
//     // #pragma omp for schedule(static, 64) nowait
//     for(i= 0 ;i < size; i+=4){
//         ptr = OriginalArray+ i;
//         complex_ptr->real = (double)*(ptr);  
//         complex_ptr->imag  = (double)*(ptr +2 ); 
//         complex_ptr++;  
//         complex_ptr->real = (double)*(ptr +1 );  
//         complex_ptr->imag  = (double)*(ptr +3 );   
//         complex_ptr++;    
//     }      
//     Complex *Reshape_ptr;
//     // change the sequence of the array, reshape it in form of Rx instead of frame
    
//     // # pragma omp for nowait
//     #pragma omp for schedule(static, 64) nowait
//     for (i = 0; i <RxSize; i++){
//         for (j = 0; j <FrameSize*ChirpSize; j++){
//             for(k =0; k< SampleSize; k++){
//                 Reshape_ptr = (Reshape+i*FrameSize*ChirpSize*SampleSize+j*SampleSize+k); 
//                 complex_ptr = (buf_complex+j*RxSize*SampleSize+i*SampleSize+k);
//                 Reshape_ptr->real = complex_ptr->real;
//                 Reshape_ptr->imag = complex_ptr->imag;
//             }
//         }
//     }
//     free(buf_complex);
//     return;
// }

// // read the file and call the "ReshapeComplex" to reshape
// int readandreshape(char *filepath, Complex *Data_reshape){
//     // ----------------------read size------------------------------
//     char filePath[] = "./fhy_direct.bin";
//     int bytesize = getBinSize(filePath);
//     int size = bytesize / sizeof(short);
//     // ----------------------read int16 ------------------------------
//     short *buf = (short*)malloc(size*sizeof(short));
//     readBin(filePath, buf, size);
//     // ----------------------reshape ------------------------------
//     short *buf_ptr = (short*)buf;
//     short *buf_reshape_real = (short*)malloc(size*sizeof(short)/2);
//     short *buf_reshape_imag  = (short*)malloc(size*sizeof(short)/2);
//     Complex *buf_reshape = (Complex *)malloc(size*sizeof(Complex)/2);
//     ReshapeComplex(buf_ptr, buf_reshape, size);
//     return size/2;
// }
