void bit()
{
    int bit;

    // 가장 낮은 1비트 제거
    int bit0 = (bit - 1) & bit;
 
    // 가장 낮은 1비트 추출
    int bit1 = (-bit0) & bit;

    
}