
#include "mkl_dfti.h"
#include "mkl.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <math.h>

using namespace sycl;
// using namespace std;

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
 

  MKL_Complex16 GetComplex(double r, double i){
    MKL_Complex16 temp;
    temp.real = r;
    temp.imag = i;
    return temp;  
  }
  
  MKL_Complex16 Complex_ADD(MKL_Complex16 a, MKL_Complex16 b){
    MKL_Complex16 temp;
    temp.real = a.real + b.real;
    temp.imag = a.imag + b.imag;
    return temp;
}


  MKL_Complex16 Complex_SUB(MKL_Complex16 a, MKL_Complex16 b){
    MKL_Complex16 temp;
    temp.real = a.real - b.real;
    temp.imag = a.imag - b.imag;
    return temp;
  }

  MKL_Complex16 Complex_MUL(MKL_Complex16 a, MKL_Complex16 b){
    MKL_Complex16 temp;
    temp.real = a.real * b.real - a.imag * b.imag;
    temp.imag = a.real * b.imag + a.imag * b.real;
    return temp;
  }


MKL_Complex16 Complex_CDiv(MKL_Complex16 a, int b){
    MKL_Complex16 temp;
    temp.real = a.real /b;
    temp.imag = a.imag /b;
    return temp;
}

void Complex_matrixMUL(MKL_Complex16 *M_res, MKL_Complex16 *M_A, MKL_Complex16 *M_B, int sizea, int sizeb, int sizec){   
    // M_A = a*b
    // M_B = b*c
    // M_res = a*c
    MKL_Complex16 tmp;
    // printf("Hi\n");
    for(int i =0;  i< sizea; i++){
        for(int k =0; k< sizec; k++){
            tmp = GetComplex(0,0);
            for(int j =0;  j< sizeb; j++){    
                tmp = Complex_ADD(Complex_MUL((*(M_A + i*sizeb + j)),(*(M_B + j*sizec + k))), tmp);
            }
            *(M_res + i*sizec + k) = tmp;
        }
    }
}

  double Complex_mol(MKL_Complex16 *a){
    return sqrt(a->real*a->real+a->imag*a->imag);
  }

int FindAbsMax(MKL_Complex16 *ptr, int size){
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

void Matrix_Transpose(MKL_Complex16 *M[], MKL_Complex16 *M_res[], int sizeRow, int sizeCol){
    for (int i = 0; i < sizeRow; i++){
        for (int j =0; j < sizeCol; j++){
            M_res[i][j] = M[j][i];
            // *(M_res + i*sizeCol + j) = *(M + j* sizeRow + i);
        }
    }
}

void printComplex(MKL_Complex16 *Data, int size){
    printf("Print!\n");
    for (int i = 0; i < size; i++){
      printf("data %d = %lf+i*%lf\n", i, (Data+i)->real,  (Data+i)->imag);
    }
}


// reshape the input bin file
// Input: *OriginalArray: the input of short bin file, size: the real size in form of short
// Output: *Reshape: reshape the input in form of complex
void ReshapeComplex(short *OriginalArray, MKL_Complex16 *Reshape, int size){
    int i, j, k;
    //int cnt = 0;
    MKL_Complex16 *buf_complex = (MKL_Complex16 *)malloc(size*sizeof(MKL_Complex16)/2);
    short *ptr;
    MKL_Complex16 *complex_ptr = buf_complex;
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
    MKL_Complex16 *Reshape_ptr;
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

void oneAPI_FFT(MKL_Complex16 *input, MKL_Complex16 *output, int size){

    DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
    // DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
    MKL_LONG status;

    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE,
                              DFTI_COMPLEX, 1, size);
    status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiComputeForward(my_desc1_handle, input, output);
    // status = DftiComputeBackward(my_desc1_handle, c2c_temp, c2c_output);
    status = DftiFreeDescriptor(&my_desc1_handle);

}

void oneAPI_FFT_inplace(MKL_Complex16 *data, int size){

    DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
    // DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
    MKL_LONG status;

    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE,
                              DFTI_COMPLEX, 1, size);
    // status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiComputeForward(my_desc1_handle, data);
    // status = DftiComputeBackward(my_desc1_handle, c2c_temp, c2c_output);
    status = DftiFreeDescriptor(&my_desc1_handle);

}


void oneAPI_2dFFT(MKL_Complex16 *input, MKL_Complex16 *output, int sizeRow, int sizeCol){

    DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
    // DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
    MKL_LONG status;
    MKL_LONG dim_sizes[2] = {sizeRow, sizeCol};

    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_SINGLE,
                              DFTI_COMPLEX, 2, dim_sizes);
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiComputeForward(my_desc1_handle, input, output); 
    status = DftiFreeDescriptor(&my_desc1_handle);
    

}

//FFTshift:
// use swap here, it need to judge whether the size is an even or an odd.
void FFTshift(MKL_Complex16 *array, int size){
    int mid = size/2;
    MKL_Complex16 tmp; 
    if(size % 2 == 0){
        for (int i = 0 ; i< mid; i++){
            tmp = *(array+ mid +i);
            *(array + mid +i) = *(array + i);
            *(array + i) = tmp;
        } 
    }
    else{
        for (int i = 0 ; i< mid; i++){
            tmp = *(array+ mid + i +1);
            *(array + mid + i + 1 ) = *(array + i);
            *(array + i) = tmp;
        } 
    } 
}

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
    auto SampleSize_extend = 128;
    auto ChirpSize_extend = ChirpSize; // getextendsize(ChirpSize);
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
    // vector<MKL_Complex16> DataHost_Frm0_reshape(size);
    // vector<double> Dis(89);
    // short(* DataHost_Frm0_read) = new short[size];
    // short(* DataHost_Frm_read) = new short[size];
    // MKL_Complex16(* DataHost_Frm0_reshape) = new MKL_Complex16[size];
    // MKL_Complex16(* DataHost_Frm_reshape) = new MKL_Complex16[size];
    // implicit USM
    short *DataShared_Frm0_read = malloc_shared<short>(ChirpSize*SampleSize*RxSize*2, q);
    short *DataShared_Frm_read = malloc_shared<short>(ChirpSize*SampleSize*RxSize*2, q);
    MKL_Complex16 *DataShared_Frm0_reshape = malloc_shared<MKL_Complex16>(size, q);
    MKL_Complex16 *DataShared_Frm_reshape = malloc_shared<MKL_Complex16>(size, q);
    // MKL_Complex16 *DataShared_Frm = malloc_shared<MKL_Complex16>(size, q);
    MKL_Complex16 *DataShared_Rx = malloc_shared<MKL_Complex16>(ChirpSize*SampleSize, q);
    MKL_Complex16 *DataShared_Rx_extended = malloc_shared<MKL_Complex16>(extendSize, q);
    MKL_Complex16 *DataShared_FFT = malloc_shared<MKL_Complex16>(extendSize, q);
    // MKL_Complex16 *DataShared_2dFFT = malloc_shared<MKL_Complex16>(extendSize, q);
    MKL_Complex16 *DataShared_2dFFT[ChirpSize];
    MKL_Complex16 *DataShared_2dFFT_tp[SampleSize_extend];
    for(int i = 0; i <ChirpSize_extend; i++){
      DataShared_2dFFT[i] = malloc_shared<MKL_Complex16>(SampleSize_extend, q);
    }
    for(int i = 0; i < SampleSize_extend; i++){
      DataShared_2dFFT_tp[i] = malloc_shared<MKL_Complex16>(ChirpSize_extend, q);
    }
    // MKL_Complex16 *DataShared_2dFFT_tp = malloc_shared<MKL_Complex16>(extendSize, q);
    // MKL_Complex16 *DataShared_chunkFFT = malloc_shared<MKL_Complex16>(128, q);
    // int *FFT_indexlist = malloc_shared<int>(extendSize, q);
    double *Dis = malloc_shared<double>(89, q);
    double *Spd = malloc_shared<double>(89, q);
    double *Agl = malloc_shared<double>(89, q);
    double *SDis = malloc_shared<double>(89, q);
    int *maxDisIdx = malloc_shared<int>(89, q);
    int *maxSpdIdx = malloc_shared<int>(89, q);
    int *maxSDisIdx = malloc_shared<int>(89, q);
    int *maxAglIdx = malloc_shared<int>(89, q);
    
    // initial the matrix
    
    int agl_sampleNum = 180/0.1+1;
    double theta;
    MKL_Complex16 *Agl_matrix = malloc_shared<MKL_Complex16>(4*agl_sampleNum, q);
    MKL_Complex16 *Agl_mulRes = malloc_shared<MKL_Complex16>(4*agl_sampleNum, q);
    // MKL_Complex16 *Agl_matrix = (MKL_Complex16 *)malloc(4*agl_sampleNum*sizeof(MKL_Complex16));
    // MKL_Complex16 *Agl_mulRes = (MKL_Complex16 *)malloc(agl_sampleNum*sizeof(MKL_Complex16));
    for(int loc=0; loc<4; loc++){
        for(int phi=-900; phi<= 900; phi++){
            theta = -loc*2*pi*d*sin((double)phi/1800.0*pi)/lamda;
            *(Agl_matrix+loc*agl_sampleNum+(phi+900)) = 
            GetComplex(cos(theta),
            sin(theta));
            
        }
    }



    // get the data
    fread(DataShared_Frm0_read, sizeof(short), ChirpSize*SampleSize*RxSize*2, infile);
    ReshapeComplex(DataShared_Frm0_read, DataShared_Frm0_reshape, readsize);

    // buffer
//     buffer<short, 1> buf_Frm0_read(DataHost_Frm0_read, size);
//     buffer<short, 1> buf_Frm_read(DataHost_Frm_read, size);
    //buffer<short, 1> buf_Frm0_read(DataHost_Frm0_read, size);

    // q.memcpy(DataDevc_Frm0, DataDecv_Frm0, sizeof(MKL_Complex16) * size).wait();

    int frm = 0;
    while(1){
        if(frm == 60 )break;
        frm++;
        // q.memcpy(DataDevc_Frm, DataDecv_Frm,, sizeof(MKL_Complex16) * size).wait();
        
        // read the data 
        if(fread(DataShared_Frm_read, sizeof(short), ChirpSize*SampleSize*RxSize*2, infile) == 0){
            break;
        }
        // reshape 
        ReshapeComplex(DataShared_Frm_read, DataShared_Frm_reshape, readsize);
        //buffer<MKL_Complex16, 1> buf_Frm0_reshape(DataHost_Frm0_reshape, size);

        
        // ******************************************************** distance *****************************************************************//            
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


        oneAPI_FFT(DataShared_Rx_extended, DataShared_FFT, extendSize);

        int maxidx = FindAbsMax(DataShared_FFT, floor(0.4*extendSize));
        // int extendSize = getextendsize(ChirpSize*SampleSize);
        double Fs_extend = Fs * extendSize/(ChirpSize*SampleSize);
        // int maxidx = FindAbsMax(DataShared_FFT, floor(0.4*extendSize));
        int maxDisidx = maxidx*(ChirpSize*SampleSize)/extendSize;
        double maxDis = cspd*(((double)maxDisidx/extendSize)*Fs_extend)/(2*mu);
        Dis[frm] = maxDis;
        maxDisIdx[frm] = maxDisidx;
        // ******************************************************** distance *****************************************************************//
        // ******************************************************** speed *****************************************************************//

        for(int chp = 0; chp < ChirpSize_extend; chp++){
            // do the 1dfft for the data in rx0
            MKL_Complex16 *chp_ptr = DataShared_Rx + chp * SampleSize;
            auto e = q.submit([&](handler& h) {
            // Preprocess: minus the frm0 data
            h.parallel_for(range<1>(SampleSize_extend), [=](id<1> i) { 
                if(i < SampleSize ){
                  DataShared_2dFFT[chp][i] = DataShared_Rx[i+chp*SampleSize];
                }
                else{
                  DataShared_2dFFT[chp][i] = GetComplex(0, 0);
                }
            });
            });
            // MKL_Complex16 *fft1dres_ptr = DataShared_2dFFT+chp * SampleSize_extend;
            // do fft for the data in Data_frm_rx0, the result is stored in Data_2dfft
            // a new version of FFTextend_OMP but divide N here:
            oneAPI_FFT_inplace(DataShared_2dFFT[chp],  SampleSize_extend);

        }

        // 2dfft: do the fft again for each column for the 1dfft result
        // ||||
        // ||||
        // VVVV
        int chp_mid = ChirpSize_extend/2 ;
        int smp_mid = SampleSize_extend/2;
        // transpose the matrix
        Matrix_Transpose(DataShared_2dFFT, DataShared_2dFFT_tp, ChirpSize_extend, SampleSize_extend);
        for(int smp = 0; smp < SampleSize_extend; smp++){
            // do the 1dfft for the data in rx0
            // MKL_Complex16 *smp_ptr = DataShared_2dFFT_tp + smp * ChirpSize_extend;
    
            oneAPI_FFT_inplace(DataShared_2dFFT_tp[smp], ChirpSize_extend);
            // fft shift, exchange the data before mid and after mid
            FFTshift(DataShared_2dFFT_tp[smp], ChirpSize_extend);
            // set the mid to 0
            DataShared_2dFFT_tp[smp][chp_mid] = GetComplex(0, 0);
            // *(smp_ptr + chp_mid) = GetComplex(0, 0);
        }
        // 2dFFT
        // for (int i  = 0 ; i< 127; i++){
        //   for(int j = 0; j< 127; j++){
        //     DataShared_2dFFT[i][j] = GetComplex(0, 0);
        //   }
        // }

        // q.submit([&](handler& h) {
        // // Preprocess: minus the frm0 data
        // h.parallel_for(range<1>(extendSize), [=](id<1> i) {
        //   int row = i / SampleSize_extend;
        //   int col = i % SampleSize_extend;
        //   int idx = row * SampleSize + col ;
        //   if(col < SampleSize ){
        //     // DataShared_2dFFT[row][col] = DataShared_Rx[idx];
        //     DataShared_2dFFT[row][col].real = row ;
        //     DataShared_2dFFT[row][col].imag = col ;
        //   }
        //   else{
        //     DataShared_2dFFT[row][col] = GetComplex(0, 0);
        //   }
        // });
          
        //     });
            // MKL_Complex16 *fft1dres_ptr = DataShared_2dFFT+chp * SampleSize_extend;
            // // do fft for the data in Data_frm_rx0, the result is stored in Data_2dfft
            // // a new version of FFTextend_OMP but divide N here:
            // oneAPI_FFT(DataShared_chunkFFT, fft1dres_ptr, SampleSize_extend);
        // for (int i = 0; i < 1; i++)
        // for (int j = 0; j < 128; j++)
        //     std::cout << i << ' '
        //               << DataShared_2dFFT[i][j].real << "+j" << DataShared_2dFFT[i][j].imag << "  "
        //               << DataShared_2dFFT[i+1][j].real << "+j" << DataShared_2dFFT[i+1][j].imag << "  "
        //               // << DataShared_2dFFT[i][100].real << "+j" << DataShared_2dFFT[i][100].imag
        //               << "\n ";
        int max2dfft =FindAbsMax(DataShared_2dFFT_tp[0],  ChirpSize * SampleSize);
        // TODO: make sure the sequence of this is totally correct
        // col
        int maxSDisidx = FindAbsMax(DataShared_2dFFT_tp[0], ChirpSize * SampleSize) / ChirpSize;
        // row
        int maxSpdidx = FindAbsMax(DataShared_2dFFT_tp[0], ChirpSize * SampleSize) % ChirpSize;
        // int maxSpdidx = FindAbsMax(Data_2dfft, ChirpSize * SampleSize) % ChirpSize;
        // int maxSDisidx = FindAbsMax(Data_2dfft, ChirpSize * SampleSize) / ChirpSize;
        double maxSpd = ((maxSpdidx)*fr/ChirpSize - fr/2)*lamda/2;
        double maxSDis = ((maxSDisidx)*Fs/SampleSize )*cspd/(2*mu);
        // the whole formula  maxDis = c*(((double)maxDisidx*Fs/(ChirpSize*SampleSize)))/(2*mu);
        Spd[frm] = maxSpd ;
        maxSpdIdx[frm] = maxSpdidx;
        SDis[frm] = maxSDis ;
        maxSDisIdx[frm] = maxSDisidx;

        // ******************************************************** speed *****************************************************************//
        // ******************************************************** Angle *****************************************************************//
        // int maxAglidx = maxidx;
        // MKL_Complex16 Agl_weight[4];
        // // fft result / length
        // // TODO: you can directly add the division of N in the FFT_extend function.
        // Agl_weight[0] = GetComplex(DataShared_FFT[maxAglidx].real/extendSize, DataShared_FFT[maxAglidx].imag/extendSize);

        // for (int j= 1; j < RxSize; j++){
        //     auto e = q.submit([&](handler& h) {
        //     // Preprocess: minus the frm0 data
        //     h.parallel_for(range<1>(extendSize), [=](id<1> i) { 
        //         if(i < SampleSize * ChirpSize ){
        //           DataShared_Rx[i] = Complex_SUB(DataShared_Frm_reshape[i+j*ChirpSize*SampleSize], DataShared_Frm0_reshape[i]);
        //           DataShared_Rx_extended[i] = DataShared_Rx[i];
        //         }
        //         else{
        //           DataShared_Rx_extended[i] = GetComplex(0, 0);
        //         }
                
        //     });
        // });
        //     // do FFT in for rx0, rx1, rx2
        //     oneAPI_FFT(DataShared_Rx_extended, DataShared_FFT, extendSize);
        //     // get the max data with the maxidx we get in the DisDetection code.
        //     Agl_weight[j] = Complex_CDiv(*(DataShared_FFT + maxAglidx), extendSize);
        // }

        // //--------------- MMM ----------------- 
        // Complex_matrixMUL(Agl_mulRes, Agl_weight, Agl_matrix, 1, 4, agl_sampleNum);
        
        // //--------------- find Max -----------------
        // maxAglidx = FindAbsMax(Agl_mulRes, agl_sampleNum);
        // double maxAgl = (maxAglidx-900.0)/10.0;
        // double maxPhi = (maxAglidx-900.0)/10.0/180*pi;
        // Agl[frm] = maxAgl ;
        // maxAglIdx[frm] = maxAglidx;

        // ******************************************************** Angle *****************************************************************//
    }
    
    //# print output
    // printComplex(DataShared_Frm_reshape, 10);
    // std::cout << frm << "\n"; 
    for (int i = 0; i < frm; i++)
     std::cout << i <<" maxDisidx "<< maxDisIdx[i] << " Dis " << Dis[i]
                    <<" maxAglidx "<< maxAglIdx[i] << " Agl " << Agl[i]
                    <<" maxSpdidx "<< maxSpdIdx[i] << " Spd " << Spd[i]
                    <<" maxSDisidx "<< maxSDisIdx[i] << " SDis " << SDis[i]
                    << "\n ";
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
    //  std::cout <<"data_Rx_extend "<< i << " " << DataShared_Rx_extended[i].real<< " " << DataShared_Rx_extended[i].imag << "\n ";
 
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
    free(DataShared_FFT);
    for(int i = 0; i <ChirpSize; i++){
      free(DataShared_2dFFT[i]);
    }
     for(int i = 0; i <ChirpSize; i++){
      free(DataShared_2dFFT_tp[i]);
    }
    free(Dis, q);
    free(Agl, q);
    free(Spd, q);
    free(SDis, q);
    free(maxDisIdx, q);
    free(maxSpdIdx, q);
    free(maxSDisIdx, q);
    free(maxAglIdx, q);

  } catch (exception const &e) {
    std::cout << "An exception is caught \n";
    std::terminate();
  }

  std::cout << "Test successfully completed on device.\n";
  return 0;
}

// int main(){
// // sycl::queue Q(sycl::default_selector{});
// // std::cout << "Running on: "
// //             << Q.get_device().get_info<sycl::info::device::name>() << "\n";
// // float _Complex c2c_data[32];
// MKL_Complex16 c2c_input[32];
// MKL_Complex16 c2c_output[32];

// MKL_Complex16 c2c_temp[32];
// // float r2c_data[34];
// DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
// DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
// MKL_LONG status;

// /* ...put values into c2c_data[i] 0<=i<=31 */
// /* ...put values into r2c_data[i] 0<=i<=31 */

// for (int i=0; i< 32; i++){
//     c2c_input[i].real = (double)i ;
//     c2c_input[i].imag = (double)i ;
//     c2c_output[i].real = (double)0 ;
//     c2c_output[i].imag = (double)0 ;
// }
// std::cout<<"Printing input result "<<"\n";
// for (int i=0 ; i< 32; i++){
//     std::cout<<i<<" real "<<c2c_input[i].real<<" imag "<<c2c_input[i].imag<<"\n";
// }
// status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE,
//                               DFTI_COMPLEX, 1, 32);
// status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
// status = DftiCommitDescriptor(my_desc1_handle);
// status = DftiComputeForward(my_desc1_handle, c2c_input, c2c_temp);
// status = DftiComputeBackward(my_desc1_handle, c2c_temp, c2c_output);
// status = DftiFreeDescriptor(&my_desc1_handle);

// std::cout<<"Printing output result "<<"\n";
// for (int i=0; i< 32; i++){
//     std::cout<<i<<" real "<<c2c_temp[i].real<<" imag "<<c2c_temp[i].imag<<"\n";
// }

// std::cout<<"Printing output result "<<"\n";
// for (int i=0; i< 32; i++){
//     std::cout<<i<<" real "<<c2c_output[i].real<<" imag "<<c2c_output[i].imag<<"\n";
// }
//     return 0 ;
// }
