
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <math.h>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <fstream>
#include <regex>

#include "jpegdecoder.h"
#include "TypeDef.h"
#include "jpegencoderMultipleQF2.h"

// for time
#include <chrono>


#define IS_PARALLEL 1

using namespace std;
using namespace cv;



double getPSNR3(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    
    Scalar s = sum(s1);        // sum elements per channel
    
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

#if IS_PARALLEL

int main(int argc, char** argv) {
    
    
    if(argc < 4)
    {
        // Tell the user how to run the program
        std::cerr << "Number of arguments should be 4: <jpegfile1> <jpegfile2> <out_file>" << std::endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    
    std::string filename = argv[1];
    // Ouptut folder path:
    std::string enc_path_to_files = argv[2];
    // out txt path:
    std::string enc_path_txt = argv[3];
    
    
//    std::string filename = "/Volumes/MULTICOM-104/validation_original/shard-0/1/ILSVRC2012_val_00000001.JPEG";
//    std::string enc_path_to_files = "/Volumes/MULTICOM-104/validation_generated_QF";
//    std::string filename = "/Users/ahamsala/Documents/validation_original/shard-0/1/ILSVRC2012_val_00000001.JPEG";
//    std::string enc_path_to_files = "/Users/ahamsala/Documents/validation_generated_QF";
//    std::string txt_path = "/Volumes/MULTICOM-104/validation_generated_QF_TXT_1";
    
    // Quality factor experiment:
    ////////////////////////////////////////////////////////
    
    try {
//"/Users/ahamsala/Documents/validation_original/shard-0/2/ILSVRC2012_val_00001777.JPEG"
//        std::string filename = "/Users/ahamsala/Documents/validation_original/shard-0/2/ILSVRC2012_val_00001777.JPEG";
//        std::string enc_path_to_files = "/Users/ahamsala/Documents/validation_generated_QF";
//        std::string enc_path_txt = "/Volumes/MULTICOM-104/validation_generated_QF_TXT_1";
//
//        std::string filename = "/Volumes/MULTICOM-104/validation_original/shard-0/1/ILSVRC2012_val_00001777.JPEG";
//        std::string enc_path_to_files = "/Volumes/MULTICOM-104/validation_generated_QF";
//        std::string enc_path_txt = "/Volumes/MULTICOM-104/validation_generated_QF_TXT_1";
//        runEncoderWithMultipleQF(filename, enc_path_to_files, enc_path_txt);
        
// 0-1 ILSVRC2012_val_00000034
// 0-2 ILSVRC2012_val_00001837
// 0-1 ILSVRC2012_val_00000001
//        string filename = "/Volumes/MULTICOM-104/validation_original/shard-2/28/ILSVRC2012_val_00027301.JPEG";
//        string filename = "/Volumes/MULTICOM-104/validation_original/shard-0/1/ILSVRC2012_val_00000001.JPEG";
//        string filename = "/Volumes/MULTICOM-104/validation_original/shard-0/2/ILSVRC2012_val_00001837.JPEG";
        
//        string filename = "/Volumes/MULTICOM-104/validation_original/shard-0/1/ILSVRC2012_val_00000034.JPEG";
//        string filename2 ="/Volumes/MULTICOM-104/validation_generated_QF/shard-0/1/ILSVRC2012_val_00000034-QF-0.JPEG";
        
//        string enc_path_to_files = "/Volumes/MULTICOM-104/validation_generated_QF";
//        string enc_path_txt = "/Volumes/MULTICOM-104/validation_generated_QF_TXT_1";
//        runEncoderWithMultipleQF(filename, enc_path_to_files, enc_path_txt);
        runEncoderWithMultipleQF(filename,enc_path_to_files, enc_path_txt);
////        runEncoderWithMultipleQF(filename, enc_path_to_files, enc_path_txt);
//
//        // Sequential
////        string filename2 = "/Volumes/MULTICOM-104/validation_generated_QF/shard-0/2/ILSVRC2012_val_00001837-QF-0.JPEG";
//        string filename2 = "/Volumes/MULTICOM-104/validation_generated_QF/shard-0/1/ILSVRC2012_val_00000001-QF-0.JPEG";
//
//        // Y test:
//
//        cout << "SEQUENTIAL TEST ----- " << endl;
//
////        Mat image_org = imread(filename, IMREAD_COLOR);
//        Mat image_org = imread(filename, IMREAD_UNCHANGED);
//        Mat image_qf = imread(filename2, IMREAD_UNCHANGED);
//        int n_comp = image_org.channels();
//        cout << "nChannels: " << n_comp << endl;
//        Mat ycbcr_org;
//        if (n_comp>1)
//        {
//            cv::cvtColor(image_org, ycbcr_org, cv::COLOR_BGR2YCrCb);
//        }
//        else
//        {
//            cout<< getPSNR(image_org, image_qf, image_org.cols, image_org.rows) <<endl;
////            imshow("Image 1", image_org);
////            waitKey(0);
//        }
//        cout << "nChannels**: " << ycbcr_org.channels() << endl;
//        cv::Mat ycbcr_channels[3];
//        cv::split(ycbcr_org, ycbcr_channels);
//
////        Mat image_qf = imread(filename2, IMREAD_COLOR);
//
//        Mat ycbcr_qf;
//        cv::cvtColor(image_qf, ycbcr_qf, cv::COLOR_BGR2YCrCb);
//        cv::Mat ycbcr_channels_qf[3];
//        cv::split(ycbcr_qf, ycbcr_channels_qf);
//
//
//        double val = getPSNR(ycbcr_channels[0], ycbcr_channels_qf[0], ycbcr_org.cols, ycbcr_org.rows);
//        cout << "FINAL PSNR: " << val << endl;
//        val = getPSNR(ycbcr_channels[1], ycbcr_channels_qf[1], ycbcr_org.cols, ycbcr_org.rows);
//        cout << "FINAL PSNR: " << val << endl;
//        val = getPSNR(ycbcr_channels[2], ycbcr_channels_qf[2], ycbcr_org.cols, ycbcr_org.rows);
//        cout << "FINAL PSNR: " << val << endl;
        
        
        
    } catch (Exception e) {
        cerr << "Input the folder properly" << endl;
    }
    return 0;
// Segment ID expected,
}

#else

int main(int argc, char** argv) {

    // Input file:
    //    std::string f1_yuv = argv[1];
    std::string f1_yuv = "/Users/ahamsala/Documents/validation_original/shard-0/2/ILSVRC2012_val_00001777.JPEG";
    
    // Input file:
    //    std::string f2_yuv = argv[2];
//    std::string f2_yuv = "/Volumes/MULTICOM-104/validation_generated_QF/shard-0/1/ILSVRC2012_val_00000001-QF-0.JPEG";
    std::string f2_yuv = "/Users/ahamsala/Documents/validation_generated_QF/shard-0/2/ILSVRC2012_val_00001777-QF-0.JPEG";
    // Output file
    //    std::string output_file = ;
    
    // first and Second Image
    cout<<"here 1"<<endl;
    jpeg_decoder Org_YUV (f1_yuv);
    cout<<"here 2"<<endl;
    jpeg_decoder QF_YUV (f2_yuv);
    cout<<"here 3"<<endl;
    
    // Declare the needed datastructures for Y, Cb, Cr:
    vector<vector<unsigned char>> ORG_vec_Y, ORG_vec_Cb, ORG_vec_Cr;
    vector<vector<unsigned char>> QF_vec_Y, QF_vec_Cb, QF_vec_Cr;
    double psnr, psnr_Y, psnr_Cb, psnr_Cr;
    double mssim, mssim_Y, mssim_Cb, mssim_Cr;
    mssim_Y = mssim_Cb = mssim_Cr = 0;
    psnr_Y  = psnr_Cb  = psnr_Cr = 0;
    
    
    int width  = Org_YUV.upscale_width;
    int height = Org_YUV.upscale_height;
    int width2  = QF_YUV.upscale_width;
    int height2 = QF_YUV.upscale_height;
    int nComponents = Org_YUV.numberOfComponents;
    int nComponents2 = QF_YUV.numberOfComponents;
    
    assert(width  == width2);
    assert(height == height2);
    assert(nComponents == nComponents2);
    
    cout<<"width:"<<width<<"height:"<<height<<endl;
    // Create the mats
    cv::Mat ORG_mat_Y, ORG_mat_Cb, ORG_mat_Cr;
    cv::Mat QF_mat_Y, QF_mat_Cb, QF_mat_Cr;
    cv::Mat QF_mat_Cr_420, QF_mat_Cb_420, ORG_mat_Cr_420 , ORG_mat_Cb_420;
    
    // Fill the datastructures with values
    ORG_vec_Y.resize(height, vector<unsigned char> (width, 0));
    ORG_vec_Y = Arr2Vec(Org_YUV.m_YPicture_buffer, width, height);
    ORG_mat_Y = vec2mat(ORG_vec_Y);
    
    QF_vec_Y.resize(height, vector<unsigned char> (width, 0));
    QF_vec_Y = Arr2Vec(QF_YUV.m_YPicture_buffer, width, height);
    QF_mat_Y = vec2mat(QF_vec_Y);
    
    
    psnr_Y = getPSNR(ORG_mat_Y, QF_mat_Y, width, height);
    Scalar final_mssim = getMSSIM(ORG_mat_Y, QF_mat_Y);
    mssim_Y = final_mssim[0];
    
    
    if (nComponents > 1)
    {
        // Org:
        // Cb component
        ORG_vec_Cb.resize(height, vector<unsigned char> (width, 0));
        ORG_vec_Cb = Arr2Vec(Org_YUV.m_CbPicture_buffer, width, height);
        ORG_mat_Cb = vec2mat(ORG_vec_Cb);
        
        
        // Cr component
        ORG_vec_Cr.resize(height, vector<unsigned char> (width, 0));
        ORG_vec_Cr = Arr2Vec(Org_YUV.m_CrPicture_buffer, width, height);
        ORG_mat_Cr = vec2mat(ORG_vec_Cr);
        
        // QF:
        // Cb component
        QF_vec_Cb.resize(height, vector<unsigned char> (width, 0));
        QF_vec_Cb = Arr2Vec(QF_YUV.m_CbPicture_buffer, width, height);
        QF_mat_Cb= vec2mat(QF_vec_Cb);
        
        
        // Cr component
        QF_vec_Cr.resize(height, vector<unsigned char> (width, 0));
        QF_vec_Cr = Arr2Vec(QF_YUV.m_CrPicture_buffer, width, height);
        QF_mat_Cr = vec2mat(QF_vec_Cr);
        
        
        //         Calculate the quality metrics
        psnr_Cb = getPSNR(ORG_mat_Cb, QF_mat_Cb, width, height);
        psnr_Cr = getPSNR(ORG_mat_Cr, QF_mat_Cr, width, height);
        
        Scalar final_mssim = getMSSIM(ORG_mat_Cb, QF_mat_Cb);
        mssim_Cb = final_mssim[0];
        final_mssim = getMSSIM(ORG_mat_Cr, QF_mat_Cr);
        mssim_Cr = final_mssim[0];
        
        resize(ORG_mat_Cr, ORG_mat_Cr_420, Size(width/2,height/2));
        resize(ORG_mat_Cb, ORG_mat_Cb_420, Size(width/2,height/2));
        resize(QF_mat_Cr, QF_mat_Cr_420, Size(width/2,height/2));
        resize(QF_mat_Cb, QF_mat_Cb_420, Size(width/2,height/2));
        
//        imshow("ORG_mat_Cr",ORG_mat_Cr);
//        cv::waitKey(0);
//        imshow("ORG_mat_Cr",ORG_mat_Cr);
//        cv::waitKey(0);
//        imshow("QF_mat_Cb",ORG_mat_Cb);
//        cv::waitKey(0);
//        imshow("QF_mat_Cb",ORG_mat_Cb);
//        cv::waitKey(0);
        
    }
    
    FILE* pFile = fopen("/Users/ahamsala/Documents/3.a-codes/image.yuv", "wb");
    
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            fwrite( &ORG_mat_Y.at<unsigned char>(i,j), sizeof(unsigned char), 1, pFile );
        }
    }
    for(int i = 0; i < height/2; ++i) {
        for(int j = 0; j < width/2; ++j) {
            fwrite( &ORG_mat_Cb_420.at<unsigned char>(i,j), sizeof(unsigned char), 1, pFile );
        }
    }
    for(int i = 0; i < height/2; ++i) {
        for(int j = 0; j < width/2; ++j) {
            fwrite( &ORG_mat_Cr_420.at<unsigned char>(i,j), sizeof(unsigned char), 1, pFile );
        }
    }
    fclose(pFile);
    
    mssim = (6*mssim_Y + mssim_Cb + mssim_Cr)/8;
    psnr  =(6*psnr_Y + psnr_Cb + psnr_Cr)/8;
    
    reportMetrics(psnr_Y, psnr_Cb, psnr_Cr);
    reportMetrics(mssim_Y, mssim_Cb, mssim_Cr);
    
    
    
    return 0;
    
}
#endif
