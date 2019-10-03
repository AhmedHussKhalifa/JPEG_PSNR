//
//  jpegencoderMultipleQF.h
//  jpeg_tcm
//
//  Created by Hossam Amer on 2018-11-04.
//  Copyright © 2018 Hossam Amer. All rights reserved.
//

#ifndef JPEG_ENCODER_MULTIPLE_QF_H
#define JPEG_ENCODER_MULTIPLE_QF_H
#define Enable_MultiEncoder 0
#define Enable_MultiDecoder 1
#include <thread>

#include "jpegdecoder.h"
#include "jpegencoder.h"
#define S7S_debug 0
static double getPSNR(const cv::Mat& i1, const cv::Mat& i2, int comp_width, int comp_height)
{
    
    int d     = CV_8U;
    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    
    
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
#if S7S_debug
    cout<<"here **"<<endl;
    for(int i = 0; i < 8; ++i){
        for(int j = 0; j < 8; ++j){
            cout << int(s1.at<unsigned char>(i,j))<<","<<int(I1.at<unsigned char>(i,j))<<","<< int(I2.at<unsigned char>(i,j))<< " ";
        }
        cout << "\n";
    }
#endif
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    
    Scalar s = sum(s1);         // sum elements per channel
    
    double mse = (s[0]); // sum channels
#if S7S_debug
    cout << "MSE: " << mse << endl;
#endif
    if( mse <= 1e-10) // for small values return zero
    {
        return 999.99;
    }
    else
    {
        double psnr = 10.0*log10(double(255*255*double(comp_width*comp_height))/mse);
        return psnr;
    }
}

Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;
    
    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    
    /*************************** END INITS **********************************/
    
    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    
    Mat sigma1_2, sigma2_2, sigma12;
    
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    
    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;
    
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    
    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
    
    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}


// Convert double pointer to vector
vector<vector<unsigned char>> Arr2Vec(unsigned char** x, int jpegImageWidth , int jpegImageHeight)
{
    vector<vector<unsigned char>> temp;
    temp.resize(jpegImageHeight, vector<unsigned char> (jpegImageWidth, 0));
    
    for(int i = 0; i < jpegImageHeight; ++i) {
        for(int j = 0; j < jpegImageWidth; ++j) {
            temp[i][j] = x[i][j];
        }
    }
    
    return temp;
}

cv::Mat vec2mat(vector< vector<unsigned char> > &matrix){
    
    // Copy data flag has to be true
    // [https://answers.opencv.org/question/96739/returning-a-mat-from-a-function/]
    cv::Mat matrix_CV(0, matrix.size(), cv::DataType<unsigned char>::type, true);
    
    for (unsigned int i = 0; i < matrix.size(); ++i)
    {
        // Make a temporary cv::Mat row and add to NewSamples _without_ data copy
        cv::Mat Sample(1, matrix[0].size(), cv::DataType<unsigned char>::type, matrix[i].data());
        matrix_CV.push_back(Sample);
    }
    return matrix_CV;
}

void reportMetrics(double x, double y, double z)
{
    cout << "X: " << x << "\nY: " << y << "\nZ: " << z << endl;
}


// parallel running
static bool runEncoderWithMultipleQF(std::string filename, std::string enc_path_to_files, std::string txt_path) {
    
    std::string main_dir = filename.substr(0,52);
    std::string first_token = filename.substr(41);
//    std::string main_dir = filename.substr(0,46);
//    std::string first_token = filename.substr(45);
//
    size_t found1 = first_token.find_last_of("/\\");
    std::string second_token = first_token.substr(0,found1+1);
    std::string third_token = first_token.substr(found1+1);
    size_t found2 = third_token.find_last_of(".");
    std::string fifth_token = third_token.substr(found2);
    std::string fourth_token = third_token.substr(0,found2);
    
//    1 /shard-0/1/ILSVRC2012_val_00000001.JPEG
//    2 /shard-0/1/
//    3 ILSVRC2012_val_00000001.JPEG
//    4 ILSVRC2012_val_00000001
//    5 .JPEG
//    6 ILSVRC2012_val_00000001
//    std::cout <<"1 " + first_token << std::endl;
//    std::cout <<"2 " + second_token << std::endl;
//    std::cout <<"3 " + third_token << std::endl;
//    std::cout <<"4 " + fourth_token << std::endl;
//    std::cout <<"5 " + fifth_token << std::endl;
//    std::cout <<"6 " + fourth_token << std::endl;
    
    
    std::string f1_yuv = filename;
    jpeg_decoder Org_YUV(f1_yuv);
    
    int nloop = 21;
    std::vector<jpeg_decoder> qf_decoders;
    for (int i = 0; i < nloop; ++i)
    {
        qf_decoders.push_back(jpeg_decoder());
    }
    
    

    // Parallel version
    // number of threads from the given hardware
    const size_t nthreads = std::thread::hardware_concurrency();
    {
        // Pre loop
        //        std::cout <<"parallel (" << nthreads << " threads):" <<std::endl;
        std::vector<std::thread> threads(nthreads);
        std::mutex critical;
        
        // Create the threads
        for(int t = 0; t < nthreads; ++t)
        {
            threads[t] = std::thread(std::bind(
                                               [&](const int bi, const int ei, const int t)
                                               {
                                                   // loop over all items
                                                   for(int i = bi; i < ei ; i++)
                                                   {
                                                       {
                                                           const int quality_factor = i*5;
                                                           std::string decoded_filename = enc_path_to_files + second_token+ fourth_token + "-QF-" + to_string(quality_factor) + fifth_token;
                                                           cout<<decoded_filename<<endl;
                                                           qf_decoders[i].setJPEGFileNameAndStart(decoded_filename);
                                                           
                                                       }
                                                   }
                                                   
                                               },
                                               t * nloop/nthreads,
                                               (t+1)==nthreads? nloop : (t+1)*nloop/nthreads,
                                               t)
                                     );
        }
        
        
        
        // Launch the threads:
        std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
        
        
        
        // Cal
        
        vector<vector<unsigned char>> ORG_vec_Y, ORG_vec_Cb, ORG_vec_Cr;
        vector<vector<unsigned char>> QF_vec_Y, QF_vec_Cb, QF_vec_Cr;
        double psnr, psnr_Y, psnr_Cb, psnr_Cr;
        double mssim, mssim_Y, mssim_Cb, mssim_Cr;
        mssim_Y = mssim_Cb = mssim_Cr = 0;
        psnr_Y  = psnr_Cb  = psnr_Cr = 0;
        cv::Mat QF_mat_Cr_420, QF_mat_Cb_420, ORG_mat_Cr_420 , ORG_mat_Cb_420;
        
        for (int kDecoder = 0; kDecoder < nloop; ++kDecoder)
        {
            
            
            int width  = Org_YUV.upscale_width;
            int height = Org_YUV.upscale_height;
            int width2  = qf_decoders[kDecoder].upscale_width;
            int height2 = qf_decoders[kDecoder].upscale_height;
            int nComponents = Org_YUV.numberOfComponents;
            int nComponents2 = qf_decoders[kDecoder].numberOfComponents;
            
            assert(width  == width2);
            assert(height == height2);
            assert(nComponents == nComponents2);
            
            // Create the mats
            cv::Mat ORG_mat_Y, ORG_mat_Cb, ORG_mat_Cr;
            cv::Mat QF_mat_Y, QF_mat_Cb, QF_mat_Cr;
            
            
            // Fill the datastructures with values
            ORG_vec_Y.resize(height, vector<unsigned char> (width, 0));
            ORG_vec_Y = Arr2Vec(Org_YUV.m_YPicture_buffer, width, height);
            ORG_mat_Y = vec2mat(ORG_vec_Y);
            
            QF_vec_Y.resize(height, vector<unsigned char> (width, 0));
            QF_vec_Y = Arr2Vec(qf_decoders[kDecoder].m_YPicture_buffer, width, height);
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
                QF_vec_Cb = Arr2Vec(qf_decoders[kDecoder].m_CbPicture_buffer, width, height);
                QF_mat_Cb= vec2mat(QF_vec_Cb);
                
                
                // Cr component
                QF_vec_Cr.resize(height, vector<unsigned char> (width, 0));
                QF_vec_Cr = Arr2Vec(qf_decoders[kDecoder].m_CrPicture_buffer, width, height);
                QF_mat_Cr = vec2mat(QF_vec_Cr);
                
                resize(ORG_mat_Cr, ORG_mat_Cr_420, Size(width/2,height/2));
                resize(ORG_mat_Cb, ORG_mat_Cb_420, Size(width/2,height/2));
                resize(QF_mat_Cr, QF_mat_Cr_420, Size(width/2,height/2));
                resize(QF_mat_Cb, QF_mat_Cb_420, Size(width/2,height/2));
                
                //         Calculate the quality metrics
                psnr_Cb = getPSNR(ORG_mat_Cb, QF_mat_Cb, width, height);
                psnr_Cr = getPSNR(ORG_mat_Cr, QF_mat_Cr, width, height);
                
                Scalar final_mssim = getMSSIM(ORG_mat_Cb, QF_mat_Cb);
                mssim_Cb = final_mssim[0];
                final_mssim = getMSSIM(ORG_mat_Cr, QF_mat_Cr);
                mssim_Cr = final_mssim[0];
                mssim = (6*mssim_Y + mssim_Cb + mssim_Cr)/8;
                psnr  = (6*psnr_Y + psnr_Cb + psnr_Cr)/8;
            }
            else
            {
                mssim = mssim_Y;
                psnr = psnr_Y;
            }
            

            
//            reportMetrics(psnr_Y, psnr_Cb, psnr_Cr);
//            reportMetrics(mssim_Y, mssim_Cb, mssim_Cr);
            
            // Write to out file:
            const int quality_factor = kDecoder*5;
            ofstream myfile_txt;
            myfile_txt.open(txt_path + second_token+ fourth_token + "-QF-" + to_string(quality_factor) + ".txt");
            myfile_txt << to_string(psnr) << "\t" << to_string(mssim) << "\n";
            myfile_txt.close();
            
            // delete the raw pointer for this decoder
            qf_decoders[kDecoder].deleteRawPictureBufferPointers();
            
        } // end kDecoder loop
        
        // Post loop:
        // ---------
        
        
        std::string encoded_filename = filename;
        
        ////// String Processing -- Get the file Name
        size_t found = encoded_filename.find_last_of("/\\");
        std::string filename_first_token = encoded_filename.substr(found+1);
        found = filename_first_token.find_first_of(".");
        std::string filename_second_token = filename_first_token.substr(0, found);
        cout << "\n\nDone Execution; Output is: " << enc_path_to_files + second_token+ fourth_token + "-QF-* .JPEG" << endl;
        cout << "\n\nDone Execution; Output is: " << txt_path + second_token+ fourth_token + "-QF-* .txt" << endl;
    }

    return true;
    
}

// filename: input file name
// enc_path_to_files: path to store the encoded pictures
//static bool runEncoderWithMultipleQF(std::string filename, std::string enc_path_to_files, std::string txt_path, double cosine_idct[8][8])
//{
//#if Enable_MultiDecoder
//    std::string encoded_path = "/Users/ahamsala/Documents/validation_generated_QF";
//    std::string encoded_filename = filename;
//
//
//
//    //   ILSVRC2012_val_00000001-QF
//    ////// String Processing -- Get the file Name
//    //   /Users/ahamsala/Documents/validation_generated_QF/shard-0/1/ILSVRC2012_val_00000001-QF-5.JPEG
////    std::cout<<encoded_filename<<std::endl;
//    // /shard-0/1/ILSVRC2012_val_00000001.JPEG
////    std::string main_dir = encoded_filename.substr(0,52);
//    std::string main_dir = encoded_filename.substr(0,46);
////    std::cout<<"* "+main_dir<<std::endl;
//    std::string first_token = encoded_filename.substr(45);
////    std::string first_token = encoded_filename.substr(41);
////
////    std::cout<<"1 "+first_token<<std::endl;
//    size_t found1 = first_token.find_last_of("/\\");
//    std::string second_token = first_token.substr(0,found1+1);
//    // /shard-0/1/
////    std::cout<<"2 "+second_token<<std::endl;
//    //  ILSVRC2012_val_00000001.JPEG
//    std::string third_token = first_token.substr(found1+1);
//
////    std::cout<<"3 "+third_token<<std::endl;
//    // .JPEG
//    size_t found2 = third_token.find_last_of(".");
//    std::string fifth_token = third_token.substr(found2);
////    std::cout<<"5 "+fifth_token<<std::endl;
//    std::string fourth_token = third_token.substr(0,found2);
//    // ILSVRC2012_val_00000001
////    std::cout<<"4 "+fourth_token<<std::endl;
////    std::cout<<enc_path_to_files<<std::endl;
////    exit(0);
//    /*
//     /Users/ahamsala/Documents/validation_original/shard-4/41/ILSVRC2012_val_00040001.JPEG
//     1 /shard-4/41/ILSVRC2012_val_00040001.JPEG
//     2 /shard-4/41/
//     3 ILSVRC2012_val_00040001.JPEG
//     5 .JPEG
//     4 ILSVRC2012_val_00040001
//     /Volumes/MultiCom_104/validation_generated_QF/shard-0/1/ILSVRC2012_val_00000001-QF-0.JPEG
//     */
//
//
//
//#endif
//
//#if Enable_MultiEncoder
//    // Decode:
//    cout << "Start Decode " << filename << endl;
//    jpeg_decoder test(filename, cosine_idct);
////    test.display_jpg_yuv(filename);
//    cout << "Done Decode " << filename  << endl;
//
//    // Hossam: Save the input fileName
//    std::string encoded_filename = filename;
//
//    ////// String Processing -- Get the file Name
//    size_t found = encoded_filename.find_last_of("/\\");
//    std::string filename_first_token = encoded_filename.substr(found+1);
//    found = filename_first_token.find_first_of(".");
//    std::string filename_second_token = filename_first_token.substr(0, found);
//
////    std::cout<<enc_path_to_files+filename_second_token +".txt"<<std::endl;
//    ofstream myfile_txt;
//    myfile_txt.open(enc_path_to_files + filename_second_token +".txt");
//
//
//#endif
//
//    // Number of loops is 21 (21 quality factors)
//    int nloop = 21;
//
//    // Parallel version
//    // number of threads from the given hardware
//    const size_t nthreads = std::thread::hardware_concurrency();
//    {
//        // Pre loop
//        //        std::cout <<"parallel (" << nthreads << " threads):" <<std::endl;
//        std::vector<std::thread> threads(nthreads);
//        std::mutex critical;
//
//        // Create the threads
//        for(int t = 0; t < nthreads; ++t)
//        {
//            threads[t] = std::thread(std::bind(
//                                               [&](const int bi, const int ei, const int t)
//                                               {
//#if Enable_MultiDecoder
//                                                   vector<vector<unsigned char>> ORG_vec_Y, ORG_vec_Cb, ORG_vec_Cr;
//                                                   cv::Mat ORG_mat_Cr_420, ORG_mat_Cb_420;
//                                                   cv::Mat ORG_mat_Y;
//                                                   double mssim_Y,mssim_Cb,mssim_Cr,mssim;
//                                                   double psnr_Y,psnr_Cb,psnr_Cr,psnr;
//                                                   mssim_Cb = 0;
//                                                   psnr_Cb = 0;
//                                                   mssim_Cr = 0;
//                                                   psnr_Cr = 0;
//                                                   {
//                                                       // first  Image
//                                                       std::string f1_yuv = encoded_filename;
//                                                       //        cout<<"here1"<<endl;
////                                                       cout<<"1:"<<f1_yuv<<endl;
////                                                       jpeg_decoder* Org_YUV;
////                                                       Org_YUV = new jpeg_decoder(f1_yuv,cosine_idct);    // Convert Picture buffer into vector
//                                                       jpeg_decoder Org_YUV(f1_yuv,cosine_idct);
//                                                       ORG_vec_Y.resize(Org_YUV.jpegImageHeight, vector<unsigned char> (Org_YUV.jpegImageWidth, 0));
//                                                       ORG_vec_Y = Arr2Vec(Org_YUV.m_YPicture_buffer,Org_YUV.jpegImageWidth, Org_YUV.jpegImageHeight);
//                                                       ORG_mat_Y = vec2mat(ORG_vec_Y);
//                                                       //                                                       exit(0);
//                                                       if (Org_YUV.components.size() > 1)
//                                                       {
//                                                           // Cb component
//                                                           ORG_vec_Cb.resize(Org_YUV.jpegImageHeight, vector<unsigned char> (Org_YUV.jpegImageWidth, 0));
//                                                           ORG_vec_Cb = Arr2Vec(Org_YUV.m_CbPicture_buffer,Org_YUV.jpegImageWidth, Org_YUV.jpegImageHeight);
//                                                           cv::Mat ORG_mat_Cb = vec2mat(ORG_vec_Cb);
//                                                           resize(ORG_mat_Cb, ORG_mat_Cb_420, Size(Org_YUV.jpegImageWidth/2,Org_YUV.jpegImageHeight/2));
//
//                                                           // Cr component
//                                                           ORG_vec_Cr.resize(Org_YUV.jpegImageHeight, vector<unsigned char> (Org_YUV.jpegImageWidth, 0));
//                                                           ORG_vec_Cr = Arr2Vec(Org_YUV.m_CrPicture_buffer,Org_YUV.jpegImageWidth, Org_YUV.jpegImageHeight);
//                                                           cv::Mat ORG_mat_Cr= vec2mat(ORG_vec_Cr);
//                                                           resize(ORG_mat_Cr, ORG_mat_Cr_420, Size(Org_YUV.jpegImageWidth/2,Org_YUV.jpegImageHeight/2));
//                                                       }
//
//                                                   }
//                                                   // loop over all items
//                                                   for(int i = bi; i < ei ; i++)
//                                                   {
//                                                       const int quality_factor = i*5;
//                                                       // Decode:
//                                                       // inner loop
//                                                       {
////                                                           cout<<"here2"<<endl;
////                                                           jpeg_decoder* QF_YUV;
//                                                           // Decode Filenames:
//                                                           std::string decoded_filename = enc_path_to_files + second_token+ fourth_token + "-QF-" + to_string(quality_factor) + fifth_token;
////                                                           cout<<"2:"<<decoded_filename<<endl;
////                                                           QF_YUV = new jpeg_decoder(decoded_filename,cosine_idct);
//                                                           jpeg_decoder QF_YUV(decoded_filename,cosine_idct);
////                                                           exit(0);
//                                                           // Convert Picture buffer into vector
//                                                           vector<vector<unsigned char>> QF_vec_Y, QF_vec_Cb, QF_vec_Cr;
//                                                           cv::Mat QF_mat_Cr_420, QF_mat_Cb_420;
//                                                           QF_vec_Y.resize(QF_YUV.jpegImageHeight, vector<unsigned char> (QF_YUV.jpegImageWidth, 0));
//                                                           QF_vec_Y = Arr2Vec(QF_YUV.m_YPicture_buffer,QF_YUV.jpegImageWidth, QF_YUV.jpegImageHeight);
//                                                           cv::Mat QF_mat_Y = vec2mat(QF_vec_Y);
//                                                           // Calculate the quality metrics
//                                                           psnr_Y = getPSNR(ORG_mat_Y, QF_mat_Y, QF_YUV.jpegImageWidth, QF_YUV.jpegImageHeight);
//                                                           Scalar final_mssim = getMSSIM(ORG_mat_Y, QF_mat_Y);
//                                                           mssim_Y = final_mssim[0];
//
//                                                           if (QF_YUV.components.size() > 1)
//                                                           {
//                                                               // Cb component
//                                                               QF_vec_Cb.resize(QF_YUV.jpegImageHeight, vector<unsigned char> (QF_YUV.jpegImageWidth, 0));
//                                                               QF_vec_Cb = Arr2Vec(QF_YUV.m_CbPicture_buffer,QF_YUV.jpegImageWidth, QF_YUV.jpegImageHeight);
//                                                               cv::Mat QF_mat_Cb= vec2mat(QF_vec_Cb);
//                                                               resize(QF_mat_Cb, QF_mat_Cb_420, Size(QF_YUV.jpegImageWidth/2,QF_YUV.jpegImageHeight/2));
//
//                                                               // Cr component
//                                                               QF_vec_Cr.resize(QF_YUV.jpegImageHeight, vector<unsigned char> (QF_YUV.jpegImageWidth, 0));
//                                                               QF_vec_Cr = Arr2Vec(QF_YUV.m_CrPicture_buffer,QF_YUV.jpegImageWidth, QF_YUV.jpegImageHeight);
//                                                               cv::Mat QF_mat_Cr= vec2mat(QF_vec_Cr);
//                                                               resize(QF_mat_Cr, QF_mat_Cr_420, Size(QF_YUV.jpegImageWidth/2,QF_YUV.jpegImageHeight/2));
//
//                                                               // Calculate the quality metrics
//                                                               psnr_Cb = getPSNR(ORG_mat_Cb_420, QF_mat_Cb_420, QF_YUV.jpegImageWidth/2, QF_YUV.jpegImageHeight/2);
//                                                               psnr_Cr = getPSNR(ORG_mat_Cr_420, QF_mat_Cr_420, QF_YUV.jpegImageWidth/2, QF_YUV.jpegImageHeight/2);
//                                                               Scalar final_mssim = getMSSIM(ORG_mat_Cb_420, QF_mat_Cb_420);
//                                                               mssim_Cb = final_mssim[0];
//                                                               final_mssim = getMSSIM(ORG_mat_Cr_420, QF_mat_Cr_420);
//                                                               mssim_Cr = final_mssim[0];
//                                                           }
//
//
//                                                           mssim = (6*mssim_Y + mssim_Cb + mssim_Cr)/8;
//                                                           psnr =(6*psnr_Y + psnr_Cb + psnr_Cr)/8;
//                                                           std::string txt_filename = txt_path + second_token+ fourth_token + "-QF-" + to_string(quality_factor) + ".txt";
////                                                           cout<<txt_filename<<endl;
//                                                           ofstream myfile_txt;
//                                                           myfile_txt.open(txt_path + second_token+ fourth_token + "-QF-" + to_string(quality_factor) + ".txt");
////                                                           myfile_txt << to_string(psnr)+"\t"+to_string(psnr_Y)+"\t"+to_string(psnr_Cb)+"\t"+to_string(psnr_Cr)+"\t"<< to_string(mssim)+"\n";
//                                                           myfile_txt << to_string(psnr)+"\t"<< to_string(mssim)+"\n";
////                                                           cout<<to_string(psnr)+"\t"<< to_string(mssim)<<endl;
//                                                           myfile_txt.close();
//
//                                                       }
//                                                   }
//#endif
//#if Enable_MultiEncoder
//                                                   // loop over all items
//                                                   for(int i = bi; i < ei ; i++)
//                                                   {
//
//                                                       // Encode:
//                                                       // inner loop
//
//                                                       {
//                                                           const int quality_factor = i*5;
//
//                                                           // Update the full path for the encoded_file name
//                                                           encoded_filename = enc_path_to_files + filename_second_token + "-QF-" + to_string(quality_factor) + filename_first_token.substr(found);
//
//                                                           //                                                           cout << "\nStart Encode " << filename << " @ " << quality_factor << endl;
//
//                                                           jpeg_encoder enc(&test, encoded_filename, quality_factor);
//
//                                                           enc.savePicture();
//                                                           myfile_txt << filename_second_token + "-QF-" + to_string(quality_factor)+"-->"+to_string(enc.image_bpp)+"\t"<< to_string(enc.mssim)+"\t"<< to_string(enc.psnr)+"\n";
//                                                           //                                                           cout << "Done Encode; Output is " << encoded_filename << endl;
//
//                                                           // (optional) make output critical
//                                                           //                                                           std::lock_guard<std::mutex> lock(critical);
//                                                           //                                                           std::cout << bi << " " << ei << " " << quality_factor <<std::endl;
//
//                                                       }
//                                                   }
//#endif
//                                               },
//                                               t * nloop/nthreads,
//                                               (t+1)==nthreads? nloop : (t+1)*nloop/nthreads,
//                                               t)
//                                     );
//        }
//
//
//        // Launch the threads:
//        std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
//
//        // Post loop:
//        // ---------
//
//        std::string encoded_filename = filename;
//
//        ////// String Processing -- Get the file Name
//        size_t found = encoded_filename.find_last_of("/\\");
//        std::string filename_first_token = encoded_filename.substr(found+1);
//        found = filename_first_token.find_first_of(".");
//        std::string filename_second_token = filename_first_token.substr(0, found);
//        cout << "\n\nDone Execution; Output is: " << enc_path_to_files + second_token+ fourth_token + "-QF-* .JPEG" << endl;
//        cout << "\n\nDone Execution; Output is: " << txt_path + second_token+ fourth_token + "-QF-* .txt" << endl;
//    }
////    myfile_txt.close();
//    return true;
//}


#endif /* JPEG_ENCODER_MULTIPLE_QF_H */
