//
//  huffmantable.hpp
//  TCM
//
//  Created by Hossam Amer on 2018-08-09.
//  Author: Hossam Amer & Yanbing Jiang, University of Waterloo
//  Copyright © 2018 All rights reserved.
//

#ifndef HUFFMANTABLE_H
#define HUFFMANTABLE_H

#include <map>
#include <vector>
#include "TypeDef.h"

using namespace std;

class HuffmanTable {
    
public:
    
    // ID for which Huffman table it is using (there are 32 possible Huffman tables in JPEG)
    unsigned char tableID;
    
    // table length from bitstream
    unsigned short tableSegmentLengthFromBitstream;
    
    // tableClass 0 is for DC, 1 is for AC
    unsigned char  tableClass;
    vector <unsigned int> codes;
    vector <unsigned int> codeLengths;
    
    vector <unsigned char> number_of_codes_for_each_1to16;
    
    // The array of Huffman maps: (length, code) -> value
    std::map<huffKey, unsigned char> huffData;
    
    HuffmanTable();
};

#endif // HUFFMANTABLE_H
