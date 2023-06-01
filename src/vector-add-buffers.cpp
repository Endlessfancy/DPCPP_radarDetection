//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <math.h>



using namespace sycl;
// using namespace std;


// constexpr float kTol = 0.001;
// static const int num_elements = 10000;

#define SampleSize 100              // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128               // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 1                 // the frame number
#define RxSize 4                    // the rx size, which is usually 4
#define cspd 3.0e8                     // the speed of light 
#define pi 3.141592653589793               // pi 



static const double F0=77e9;                      // the initial frequency
static const double mu=5.987e12;                  // FM slope
static const double chirp_sample=100;             // the sample number in a chirp, suggesting it should be the power of 2
static const double Fs=2.0e6;                     // sampling frequency
static const double num_chirp=128;                // the chirp number in a frame, suggesting it should be the the power of 2
static const double Framenum=90;                  // the frame number 
static const double Tr=64e-6;                     // the interval of the chirp
static const double fr=1/ Tr;                     // chirp repeating frequency,
static const double lamda=cspd/F0;                   // lamda of the initial frequency 
static const double d=0.5*lamda;                  // rx_wire array distance. When it is equal to the half of the wavelength, the 
                                     // maximum unambiguous Angle can reach -90° to +90°
// static const int Tx_num = 1;
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


// class Complex {
//  private:

//  public:
//   double real, imag;

//   Complex() {
//     real = 0;
//     imag = 0;
//   }

//   Complex(int x, int y) {
//     real = x;
//     imag = y;
//   }

//   // Overloading the  != operator
//   friend bool operator!=(const Complex& a, const Complex& b) {
//     return (a.real != b.real) || (a.imag != b.imag);
//   }

//   // The function performs Complex number multiplication and returns a Complex
//   // object.
//   Complex complex_mul(const Complex& obj) const {
//     return Complex(((real * obj.real) - (imag * obj.imag)),
//                     ((real * obj.imag) + (imag * obj.real)));
//   }

//   // Overloading the ostream operator to print the objects of the Complex
//   // object
//   friend ostream& operator<<(ostream& out, const Complex& obj) {
//     out << "(" << obj.real << " : " << obj.imag << "i)";
//     return out;
//   }
// };

  // complex struct and complex algorithm
struct Complex{
    double real, imag;
};

  Complex GetComplex(double r, double i){
    Complex temp;
    temp.real = r;
    temp.imag = i;
    return temp;  
  }
  
  Complex Complex_ADD(Complex a, Complex b){
    Complex temp;
    temp.real = a.real + b.real;
    temp.imag = a.imag + b.imag;
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

void printComplex(Complex *Data, int size){
    printf("Print!\n");
    for (int i = 0; i < size; i++){
      printf("data %d = %lf+i*%lf\n", i, (Data+i)->real,  (Data+i)->imag);
    }
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
  try {
    // open file
    char filepath[] = "./fhy_direct.bin";
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
    // buffer
    // vector<short> DataHost_Frm0_read(size);
    // vector<short> DataHost_Frm_read (size);
    // vector<Complex> DataHost_Frm0_reshape(size);
    // vector<double> Dis(89);
    // short(* DataHost_Frm0_read) = new short[size];
    // short(* DataHost_Frm_read) = new short[size];
    // Complex(* DataHost_Frm0_reshape) = new Complex[size];
    // Complex(* DataHost_Frm_reshape) = new Complex[size];
    // implicit USM
    short *DataShared_Frm0_read = malloc_shared<short>(ChirpSize*SampleSize*RxSize*2, q);
    short *DataShared_Frm_read = malloc_shared<short>(ChirpSize*SampleSize*RxSize*2, q);
    Complex *DataShared_Frm0_reshape = malloc_shared<Complex>(size, q);
    Complex *DataShared_Frm_reshape = malloc_shared<Complex>(size, q);
    // Complex *DataShared_Frm = malloc_shared<Complex>(size, q);
    Complex *DataShared_Rx = malloc_shared<Complex>(ChirpSize*SampleSize, q);
    Complex *DataShared_Rx_extended = malloc_shared<Complex>(extendSize, q);
    Complex *DataShared_FFT = malloc_shared<Complex>(extendSize, q);
    int *FFT_indexlist = malloc_shared<int>(extendSize, q);
    double *Dis = malloc_shared<double>(89, q);
    int *maxDisIdx = malloc_shared<int>(89, q);
    
    
    // get the data
    fread(DataShared_Frm0_read, sizeof(short), ChirpSize*SampleSize*RxSize*2, infile);
    ReshapeComplex(DataShared_Frm0_read, DataShared_Frm0_reshape, readsize);

    // buffer
//     buffer<short, 1> buf_Frm0_read(DataHost_Frm0_read, size);
//     buffer<short, 1> buf_Frm_read(DataHost_Frm_read, size);
    //buffer<short, 1> buf_Frm0_read(DataHost_Frm0_read, size);

    // q.memcpy(DataDevc_Frm0, DataDecv_Frm0, sizeof(Complex) * size).wait();

    int frm = 0;
    while(1){
        if(frm ==1 )break;
        frm++;
        //   q.memcpy(DataDevc_Frm, DataDecv_Frm,, sizeof(Complex) * size).wait();
        // read the data 
        if(fread(DataShared_Frm_read, sizeof(short), ChirpSize*SampleSize*RxSize*2, infile) == 0){
            break;
        }
        // reshape 
        ReshapeComplex(DataShared_Frm_read, DataShared_Frm_reshape, readsize);
        //buffer<Complex, 1> buf_Frm0_reshape(DataHost_Frm0_reshape, size);

        auto e = q.submit([&](handler& h) {
            // Preprocess: minus the frm0 data
            h.parallel_for(range<1>(extendSize), [=](id<1> i) { 
                if(i < SampleSize * ChirpSize ){
                  DataShared_Rx[i] = Complex_SUB(DataShared_Frm_reshape[i], DataShared_Frm0_reshape[i]);
                  DataShared_Rx_extended[i] = DataShared_Rx[i];
                }
                else{
                  DataShared_Rx_extended[i] = GetComplex(0, 0);
                }
                
            });
        });
  
        // FFT index reverse
        q.submit([&](handler& h) {
            // int temp = 16384;
            int  l = 14;
            h.parallel_for(range<1>(extendSize), [=](id<1> i) {
                // FFT_indexlist[i] = (FFT_indexlist[i / 2] * 2) | ((i & 1)*(int)pow(2, (l - 1)));
                FFT_indexlist[i] = (FFT_indexlist[i >> 1] >> 1) | ((i & 1) << (l - 1));
            });
        });

        // swap according to the index
          q.submit([&](handler& h) {
            // Complex temp;
            h.parallel_for(range<1>(extendSize), [=](id<1> i) { 
              
                  DataShared_FFT[i] = DataShared_Rx_extended[i];
              
            });
        });
         q.submit([&](handler& h) {
            // Complex temp;
            h.parallel_for(range<1>(extendSize), [=](id<1> i) { 
                if (i < FFT_indexlist[i]) {
                  DataShared_FFT[i] = DataShared_Rx_extended[FFT_indexlist[i]];
                  DataShared_FFT[FFT_indexlist[i]] = DataShared_Rx_extended[i];
                }
            });
        });
       
        for (int mid = 1; mid < extendSize; mid <<= 1){
          Complex Wn = GetComplex(cos(pi / mid), -sin(pi / mid)); /*drop the "-" sin，then divided by len to get the IFFT*/
          for (int R = mid << 1, j = 0; j < extendSize; j += R){
            Complex w = GetComplex(1, 0);
            q.submit([&](handler& h) {
            // Preprocess: minus the frm0 data
                w = Complex_MUL(w, Wn);
                h.parallel_for(range<1>(mid), [=](id<1> k) { 
                      Complex a = DataShared_FFT[j + k];
                      Complex b = Complex_MUL(w, DataShared_FFT[j + mid + k]);
                      DataShared_FFT[j + k] = Complex_ADD(a, b);
                      DataShared_FFT[j + mid + k] = Complex_SUB(a, b);
     
                });
            });
          
        }
      }
        
        int maxidx = FindAbsMax(DataShared_Rx_extended, floor(0.4*extendSize));
        // int extendsize = getextendsize(ChirpSize*SampleSize);
        double Fs_extend = Fs * extendSize/(ChirpSize*SampleSize);
        // int maxidx = FindAbsMax(Data_fft, floor(0.4*extendSize));
        int maxDisidx = maxidx*(ChirpSize*SampleSize)/extendSize;
        double maxDis = cspd*(((double)maxDisidx/extendSize)*Fs_extend)/(2*mu);
        Dis[frm] = maxDis;
        maxDisIdx[frm] = maxDisidx;
    
    }
    
    //# print output
    // printComplex(DataShared_Frm_reshape, 10);
    // std::cout << frm << "\n"; 
    // for (int i = 0; i < 20; i++)
    //  std::cout << i <<" maxDisidx "<< maxDisIdx[i] << " Dis " << Dis[i] << "\n ";
    // for (int i = 0; i < 10; i++)
    //  std::cout <<"data_Frm0_read "<< i << " " << DataShared_Frm0_read[i] << "\n ";
    // for (int i = 0; i < 10; i++)
    //  std::cout <<"data_Frm0_reshape "<< i << " " << DataShared_Frm0_reshape[i].real<< " " << DataShared_Frm0_reshape[i].imag << "\n ";
    // for (int i = 0; i < 10; i++)
    //  std::cout <<"data_Frm_read "<< i << " " << DataShared_Frm_read[i] << "\n ";
    // for (int i = 0; i < 10; i++)
    //  std::cout <<"data_Frm_reshape "<< i << " " << DataShared_Frm_reshape[i].real<< " " << DataShared_Frm_reshape[i].imag << "\n ";
    // for (int i = 0; i < 10; i++)
    //  std::cout <<"data_Rx "<< i << " " << DataShared_Rx[i].real << " " << DataShared_Rx[i].imag << "\n ";
   
    // for (int i = 0; i < 10; i++)
    //  std::cout <<"data_FFT "<< i << " " << DataShared_FFT[i].real<< " " << DataShared_FFT[i].imag << "\n ";
    //for (int i = 0; i < frm; i++) std::cout << Dis[i] << "\n";

    fclose(infile);
    // free for the USM
    free(DataShared_Frm0_read, q);
    free(DataShared_Frm_read, q);
    free(DataShared_Frm0_reshape, q);
    free(DataShared_Frm_reshape, q);
    free(DataShared_Rx, q);
    free(DataShared_Rx_extended, q);
    free(Dis, q);
    

  } catch (exception const &e) {
    std::cout << "An exception is caught \n";
    std::terminate();
  }

  std::cout << "Test successfully completed on device.\n";
  return 0;
}

