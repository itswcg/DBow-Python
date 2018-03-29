#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
//#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
//#include <opencv2\features2d\features2d.hpp>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <map>

using namespace std;
using namespace cv;

#define input_img "the_annotated_classic_fairy_tales.jpg"
#define save_data "mydata20.txt"
#define folder "images/"
#define numImg 2
#define clusterCount 4
#define attempts 5
#define floor 8
#define des_elements 64
//typedef int DataType;

//Inverted file
struct InvertedFile {
    int num;
    vector<int> ID;
    vector<int> des_num;
    vector<float> di;
    float weight;
};

//Struct tree
struct CenterTree {
    vector<float> data; //Luu gia tri cua nut
};
//Time
double LiToDouble(LARGE_INTEGER x)
{
    double result = ((double)x.HighPart) * 4.294967296E9 + (double)((x).LowPart);
    return result;
}

double getTime()
{
    LARGE_INTEGER lpFrequency, lpPerfomanceCount;
    QueryPerformanceFrequency(&lpFrequency);
    QueryPerformanceCounter(&lpPerfomanceCount);
    return LiToDouble(lpPerfomanceCount) / LiToDouble(lpFrequency);
}
//Luy thua
long power(int x, int y)
{
    if (y==0)return 1;
    long u=power(x,y/2);
    if (y%2==0)return u*u;
    return u*u*x;
}

/*-------------------------------------------------------------------------------------------------------------------------------*/
//Phan cum k means
void kMeans(Mat& descriptors, vector<int>& labels, Mat& centers, int level, vector<Mat>& CenterArray, Mat& CenterParent) {
    if (level != 0){
        if (descriptors.rows >= clusterCount) { //So descritors ko nho hon so cum.
            kmeans(descriptors, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.00000001), attempts, KMEANS_PP_CENTERS , centers );
            CenterArray[level-1].push_back(centers);
            vector<Mat> descriptors_list(clusterCount);
            for (int i = 0; i < labels.size(); i++) {
                Mat row = descriptors.row(i);
                descriptors_list[labels[i]].push_back(row);
            }

            for (int i = 0; i < clusterCount; i++) {
                kMeans(descriptors_list[i], labels, centers, level-1, CenterArray, centers.row(i));
            }
        } else if (descriptors.rows < clusterCount) { //So descriptors nho hon cum
            int rows = clusterCount - descriptors.rows;
            kmeans(descriptors, descriptors.rows, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.00000001), attempts, KMEANS_PP_CENTERS  , centers );
            for (int i = 0; i < rows; i++) {
                centers.push_back(CenterParent);
            }
            CenterArray[level-1].push_back(centers);
            vector<Mat> descriptors_list(clusterCount);
            for (int i = 0; i < labels.size(); i++) {
                Mat row = descriptors.row(i);
                descriptors_list[labels[i]].push_back(row);
            }
            for (int i = 0; i < clusterCount; i++) {
                if (descriptors_list[i].data != 0) {
                    kMeans(descriptors_list[i], labels, centers, level-1, CenterArray, centers.row(i));
                } else {
                    kMeans(centers.row(i), labels, centers, level-1, CenterArray, centers.row(i));
                }
            }
        }
    }
}

//Khoang cach euclid
float des_distance(vector<float> &descriptors, vector<float> &center) {
    float sum = 0;
    for (int i = 0; i < des_elements; i++) {
        sum += ((descriptors[i] - center[i]) * (descriptors[i] - center[i]));
    }
    return sum;
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------*/
int descriptor_down(vector<float> &descriptor, vector<CenterTree> &MyTree) {
    int k = 0;
    int pos;
    while (clusterCount*k+1 < MyTree.size()) {
        float min_dis = 1e6;
        for (int j = 1; j <= clusterCount; j++) {
            float d = des_distance(descriptor, MyTree[clusterCount * k + j].data);
            if ( d < min_dis) {
                min_dis = d;
                pos = clusterCount * k + j;
            }
        }
        k = pos;
        /*cout<<k;getchar();*/
    }
    return pos;

}

/*-----------------------------------------------------------------------------------------------------------------------------------------------------*/
void Save_Inverted(long i_pos, vector<InvertedFile> &MyFile, long i) {
    //ifile << i_pos << endl;
    //cout << pos << endl;
    int flag = 0;
    if (!MyFile[i_pos].ID.empty()) {
        for (int j = 0; j < MyFile[i_pos].ID.size(); j++) {
            if (MyFile[i_pos].ID.at(j) == i) {
                MyFile[i_pos].des_num.at(j)++;
                flag = 1;
            } /*else {
                flag = 0;
            }*/
        }
        if (flag == 0) {
            MyFile[i_pos].ID.push_back(i);
            MyFile[i_pos].des_num.push_back(1);
        }
    } else {
        MyFile[i_pos].ID.push_back(i);
        MyFile[i_pos].des_num.push_back(1);
    }
    /*int flag = 0;
    if (MyFile[i_pos].num != 0) {
        for (int j = 0; j < MyFile[i_pos].num; j++) {
            if (MyFile[i_pos].ID.at(j) == i) {
                MyFile[i_pos].des_num.at(j)++;
                flag++;
            }
        }
        if (flag == 0) {
            MyFile[i_pos].num++;
            MyFile[i_pos].ID.push_back(i);
            MyFile[i_pos].des_num.push_back(1);
        }
    } else {
        MyFile[i_pos].num++;
        MyFile[i_pos].ID.push_back(i);
        MyFile[i_pos].des_num.push_back(1);
    }*/
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------*/
void doSomethings(Mat& descriptorsDB, vector<long>& ID) {
    fstream f_file;
    f_file.open("f_file.txt", ios_base::out | ios_base::trunc);
    fstream f_file2;
    f_file2.open("f_file2.txt", ios_base::out | ios_base::trunc);
    fstream f_weight;
    f_weight.open("weight.txt", ios_base::out | ios_base::trunc);
    fstream ifile;
    ifile.open("i_file.txt", ios_base::out | ios_base::trunc);
    fstream q_file;
    q_file.open("q_file.txt", ios_base::out | ios_base::trunc);
    fstream save_file;
    save_file.open(save_data, ios_base::out | ios_base::trunc);
    fstream filet1;
    filet1.open("filet1.txt", ios_base::out | ios_base::trunc);

    Mat CenterParent;
    int level = floor;
    int imageNum = numImg;
    long TreeSize = (power(clusterCount, floor+1)-1)/(clusterCount-1);
    long FloorSize = power(clusterCount, floor);
    cout << "FloorSize: " << FloorSize << endl;
    cout << "Kich thuoc cay: " << TreeSize << endl;
    vector<int> labels;
    Mat centers;
    vector<Mat> CenterArrayLists(level);
    vector<CenterTree> MyTree(TreeSize); //Khoang tao visual tree
    vector<InvertedFile> MyFile(FloorSize); //Khoi tao inverted file

    float kmeanstime1, kmeanstime2, downtime1, downtime2;
    kmeanstime1 = getTime();
    kMeans(descriptorsDB, labels, centers, level, CenterArrayLists, CenterParent);
    kmeanstime2 = getTime();

    cout << endl << "Kmeans time: " << kmeanstime2 - kmeanstime1 << endl;

    cout << endl << "Kick thuoc cac tang: ";
    for (int i = level-1; i >= 0; i--) {
        cout << endl << CenterArrayLists[i].rows;
    }

    long count = 1;
    for (int i = level-1; i >= 0; i--) {//Luu tam node cua cac tang
        for (int j = 0; j < CenterArrayLists[i].rows; j++) {
            for (int k = 0; k < des_elements; k++) {
                MyTree[count].data.push_back(CenterArrayLists[i].at<float>(j,k));
                save_file << MyTree[count].data[k] << " ";
            }
            count++;
            save_file << endl << endl;
        }
    }
    CenterArrayLists.clear();

    vector<vector<float>> des_data(descriptorsDB.rows);
    for(long i = 0; i < descriptorsDB.rows; i++) {
        for (int j = 0; j < 64; j++) {
            des_data[i].push_back(descriptorsDB.at<float>(i,j));
            //f_file2 << des_data[i][j] << " ";
        }
        //f_file2 << endl << endl;
    }

    cout << endl << endl << "Point 1: " << endl;
    float x = des_distance(MyTree[1].data, MyTree[2].data);
    cout << "Haha:     " << x << endl;

    float temp_dis = (power(clusterCount, floor)-1) / (clusterCount-1);
    cout << "Temp distance: " << temp_dis << endl;
    /*----------------------------------------------------------------------------------------------------------------------------------------------------*/
    float d_pos;
    downtime1 = getTime();
    for (long i = 0; i < descriptorsDB.rows; i++) {
        d_pos = descriptor_down(des_data[i], MyTree);
        Save_Inverted(d_pos - temp_dis, MyFile, ID[i]);
    }
    downtime2 = getTime();
    cout << endl << "Down time: " << downtime2 - downtime1 << endl;
    cout << "FloorSize: " << FloorSize << endl;

    for (float i = 0; i < FloorSize; i++) {//Xac dinh weight
        if (MyFile[i].ID.size() != 0) {
            MyFile[i].weight = (log((float)imageNum/MyFile[i].ID.size()));
            //ifile << i << " " << MyFile[i].mymap.size() << endl;
        } else {
            MyFile[i].weight = 0;
        }
        //f_weight << MyFile[i].weight << endl;
        //save_tree << weight[i] << " ";
    }
    /*----------------------------------------------------------------------------------------------------------------------------------------------------*/

    vector<float> ch(numImg);
    for (float i = 0; i <FloorSize; i++) {
        if (MyFile[i].ID.size()>0) {
            MyFile[i].di.resize(MyFile[i].ID.size());
            for (int j = 0; j < MyFile[i].ID.size(); j++) {
                MyFile[i].di[j] = MyFile[i].des_num[j] * MyFile[i].weight;
                ch[MyFile[i].ID[j]] += MyFile[i].di[j] * MyFile[i].di[j];
            }
        }
    }

    /*for (int i = 0; i < numImg; i++) {
        cout << ch[i] << endl;
    }*/

    for (int i = 0; i < numImg; i++) {
        ch[i] = sqrt((float)ch[i]);
    }

    for (int i = 0; i <FloorSize; i++) {
        if (MyFile[i].ID.size()>0) {
            for (int j = 0; j < MyFile[i].ID.size(); j++) {
                MyFile[i].di[j] = MyFile[i].di[j]/ch[MyFile[i].ID[j]];
            }
        }
    }

    for (int i = 0; i < FloorSize; i++) {
        save_file << MyFile[i].ID.size() <<  endl;
        save_file << MyFile[i].weight << endl;
        if (MyFile[i].ID.size()>0) {
            for (int j = 0; j < MyFile[i].ID.size(); j++) {
                save_file << MyFile[i].ID[j] << " " << MyFile[i].des_num[j] << " " << MyFile[i].di[j] << " ";
            }
        }
        save_file << endl;
    }

    cout << endl << "Done chuan hoa!" << endl;

    /*--------------------------------------------------------------------------------------------------------------------------------------------------*/

    float mytime1, mytime2, mytime3, mytime4, mytime5, mytime6, mytime7, mytime8, mytime9;
    mytime1 = getTime();
    Mat queryImg = imread(input_img, 0);
    //Mat queryImg = imread("killing_yourself_to_live.jpg", 0);
    SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints;

    mytime2 = getTime();

    detector.detect(queryImg, keypoints);
    SurfDescriptorExtractor extractor;

    mytime3 = getTime();

    Mat qDescriptors;
    extractor.compute(queryImg, keypoints, qDescriptors);

    mytime4 = getTime();

    map<int, float> myMap;//need
    float pos;
    vector<float> qnum(FloorSize, 0);
    vector<vector<float>> des_query(qDescriptors.rows);

    cout << "Query descriptors: " << qDescriptors.rows;
    long qDescriptors_rows = qDescriptors.rows;

    for (int i = 0; i < qDescriptors_rows; i++) {
        for (int j = 0; j < des_elements; j++) {
            des_query[i].push_back(qDescriptors.at<float>(i,j));
            //f_file2 << des_query[i][j] << " ";
        }
        //f_file2 << endl << endl;
    }

    mytime5 = getTime();

    for (int i = 0; i < qDescriptors_rows; i++) {//Truot lan 1
        qnum[descriptor_down(des_query[i],MyTree) - temp_dis]++;
    }

    mytime6 = getTime();

    float sum_q = 0;
    for (int i = 0; i < FloorSize; i++) {
        if (qnum[i] != 0) {
            qnum[i] = qnum[i] * MyFile[i].weight;
            sum_q += qnum[i] * qnum[i];
            //ifile << qnum[i] << endl;
        }
    }
    //cout << endl << "sum_q: " << sum_q << endl;

    mytime7 = getTime();

    sum_q = sqrt((float)sum_q);
    for (int i = 0; i < FloorSize; i++) {
        if (qnum[i] != 0 && MyFile[i].ID.size() != 0) {
            qnum[i] = qnum[i]/sum_q;
            for (int j = 0; j < MyFile[i].ID.size(); j++) {
                if (myMap.find(MyFile[i].ID[j]) != myMap.end()) {
                    myMap[MyFile[i].ID[j]] += MyFile[i].di[j] * qnum[i];
                } else {
                    myMap[MyFile[i].ID[j]] = MyFile[i].di[j] * qnum[i];
                }
            }
        }
    }

    mytime8 = getTime();

    float maxv = 0;
    int possition = 0;
    for (map<int, float>::iterator ii = myMap.begin(); ii != myMap.end(); ii++) {
        //q_file << (*ii).first << "  " << (*ii).second << endl;

        if ((*ii).second > maxv) {
            maxv = (*ii).second;
            possition = (*ii).first;
        }
    }

    mytime9 = getTime();

    cout << endl << "Time read image: " << mytime2 - mytime1;
    cout << endl << "Time ex keypoints: " << mytime3 - mytime2;
    cout << endl << "Time ex descriptors: " << mytime4 - mytime3;
    cout << endl << "Time tao linh tinh: " << mytime5 - mytime4;
    cout << endl << "Time truot lan 1: " << mytime6 - mytime5;
    cout << endl << "Time xac dinh tong binh phuong: " << mytime7 - mytime6;
    cout << endl << "Time dung map: " << mytime8 - mytime7;
    cout << endl << "Time tim anh tuong tu: " << mytime9 - mytime8;
    cout << endl << "Time query: " << mytime9 - mytime5;
    cout << endl << "Tong time: " << mytime6 - mytime2;
    cout << endl << "Anh can tim: " << possition << " with score " << myMap[possition] << endl;

    /*---------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
    f_file.close();
    f_file2.close();
    f_weight.close();
    ifile.close();
    save_file.close();
    q_file.close();
    filet1.close();
}
/*----------------------------------------------------------------------------------------------------------------------------------*/

//Extract descriptor va luu vao 1 Mat
void descriptors_array(Mat& image, Mat& descriptorDB, vector<long>& ID, int& i, vector<KeyPoint> &keypoints, SurfFeatureDetector &detector, Mat &descriptors, SurfDescriptorExtractor &extractor) {
    detector.detect(image, keypoints);
    extractor.compute(image, keypoints, descriptors);
    descriptorDB.push_back(descriptors);
    for (int j = 0; j < descriptors.rows; j++) {
        ID.push_back(i);
    }
}

void readFile() {
    string dirName = folder;
    DIR *dir;
    dir = opendir(dirName.c_str());
    string imgName;
    struct dirent *ent;
    Mat descriptorsDB;
    KeyPoint coordinate;//chua dung den
    vector<long> ID;
    //vector<int> des_array(numImg);

    SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints;
    SurfDescriptorExtractor extractor;
    Mat descriptors;
    float time_ex1, time_ex2;
    if (dir != NULL) {
        time_ex1 = getTime();
        for (int i = 0; i < numImg; i++) {
            if ((ent = readdir (dir)) == NULL) break;
            imgName = ent->d_name;
            if (imgName.compare(".")!= 0 && imgName.compare("..")!= 0) {
                string aux;
                aux.append(dirName);
                aux.append(imgName);
                cout << "ID = "<< i <<": "<<aux << endl;
                Mat image = imread(aux,0);
                descriptors_array(image, descriptorsDB, ID, i, keypoints, detector, descriptors, extractor);
                //waitKey(0);
            }
            else i--;
        }
        //waitKey(0);
        closedir (dir);
        time_ex2 = getTime();
        cout << "Thoi gian xu ly database: " << time_ex2-time_ex1 << endl;
        //cout << time2 - time1 << endl << endl;
        /*for (int i = 0; i < 64; i++) {
            cout << (float)descriptorsDB.at<float>(0,i) << " ";
        }*/
        cout << "Tong descriptors: " << descriptorsDB.rows << endl;
        doSomethings(descriptorsDB, ID);
    } else {
        cout<<"Not present!"<<endl;
    }
}
/*------------------------------------------------------------------------------------------------------------------------------------------------------*/
void go() {
    fstream file;
    file.open(save_data, ios_base::in);

    fstream w_file;
    w_file.open("w_filego.txt", ios_base::out || ios_base::trunc);

    float time1, time2;

    long TreeSize = (power(clusterCount, floor+1)-1)/(clusterCount-1);
    long FloorSize = power(clusterCount, floor);
    float temp_dis = (power(clusterCount, floor)-1) / (clusterCount-1);

    cout << FloorSize << endl;
    //vector<CenterTree> MyTree(TreeSize); //Khoang tao visual tree
    vector<InvertedFile> MyFile(FloorSize);
    //vector<vector<float>> bag;

    float **MyTree = new float*[TreeSize];

    for (long i = 0; i < TreeSize; i++) {
        MyTree[i] = new float[des_elements];
    }

    for (int i = 1; i < TreeSize; i++) {
        //MyTree[i].data.resize(des_elements);
        for (int j = 0; j < 64; j++) {
            file >> MyTree[i][j];
            //w_file << MyTree[i].data[j] << " ";
        }
        //w_file << endl;
    }
    for (int i = 1; i < TreeSize; i++) {
        for (int j = 0; j < des_elements; j++) {
            //cout << MyTree[1][j] << " ";
            w_file << MyTree[i][j] << " ";
        }
    }

    float x;
    for (int i = 0; i < FloorSize; i++) {
        file >> x;
        file >> MyFile[i].weight;
        if (x > 0) {
            for (int j = 0; j < x; j++) {
                file >> x;
                MyFile[i].ID.push_back(x);
                file >> x;
                MyFile[i].des_num.push_back(x);
                file >> x;
                MyFile[i].di.push_back(x);
            }
        }
    }

    //float mytime1, mytime2, mytime3, mytime4, mytime5, mytime6, mytime7, mytime8, mytime9;
    //mytime1 = getTime();
    //Mat queryImg = imread("office.jpg", 0);
    //SurfFeatureDetector detector(400);
    //vector<KeyPoint> keypoints;

    //mytime2 = getTime();

    //detector.detect(queryImg, keypoints);
    //SurfDescriptorExtractor extractor;

    //mytime3 = getTime();

    //Mat qDescriptors;
    //extractor.compute(queryImg, keypoints, qDescriptors);

    //mytime4 = getTime();

    //map<int, float> myMap;
    //float pos;
    //vector<int> qnum(FloorSize, 0);
    //vector<vector<float>> des_query(qDescriptors.rows);

    //mytime5 = getTime();
    //
    //cout << "Query descriptors: " << qDescriptors.rows;
    //
    //for (int i = 0; i < qDescriptors.rows; i++) {
    //  for (int j = 0; j < des_elements; j++) {
    //      des_query[i].push_back(qDescriptors.at<float>(i,j));
    //  }
    //}

    //for (int i = 0; i < qDescriptors.rows; i++) {//Truot lan 1
    //  qnum[descriptor_down(des_query[i],MyTree) - temp_dis]++;
    //}

    //mytime6 = getTime();
    //
    //float sum_q = 0;
    //for (int i = 0; i < FloorSize; i++) {
    //  if (qnum[i] != 0) {
    //      qnum[i] = qnum[i] * MyFile[i].weight;
    //      sum_q += qnum[i] * qnum[i];
    //  }
    //}
    //cout << "sum_q: " << sum_q << endl;

    //mytime7 = getTime();
    //
    //sum_q = sqrt((float)sum_q);
    //for (int i = 0; i < FloorSize; i++) {
    //  if (qnum[i] != 0 && MyFile[i].ID.size() != 0) {
    //      qnum[i] = qnum[i]/sum_q;
    //      for (int j = 0; j < MyFile[i].ID.size(); j++) {
    //          if (myMap.find(MyFile[i].ID[j]) != myMap.end()) {
    //              myMap[MyFile[i].ID[j]] += MyFile[i].di[j] * qnum[i];
    //          } else {
    //              myMap[MyFile[i].ID[j]] = MyFile[i].di[j] * qnum[i];
    //          }
    //      }
    //  }
    //}

    //mytime8 = getTime();
    //
    //float maxv = 0;
    //int possition = 0;
    //for (map<int, float>::iterator ii = myMap.begin(); ii != myMap.end(); ii++) {
    //  if ((*ii).second > maxv) {
    //      maxv = (*ii).second;
    //      possition = (*ii).first;
    //  }
    //}

    //mytime9 = getTime();
    //
    //cout << endl << "Time read image: " << mytime2 - mytime1;
    //cout << endl << "Time ex keypoints: " << mytime3 - mytime2;
    //cout << endl << "Time ex descriptors: " << mytime4 - mytime3;
    //cout << endl << "Time tao linh tinh: " << mytime5 - mytime4;
    //cout << endl << "Time truot lan 1: " << mytime6 - mytime5;
    //cout << endl << "Time xac dinh tong binh phuong: " << mytime7 - mytime6;
    //cout << endl << "Time dung map: " << mytime8 - mytime7;
    //cout << endl << "Time tim anh tuong tu: " << mytime9 - mytime8;
    //cout << endl << "Time query: " << mytime9 - mytime5;
    //cout << endl << "Tong time: " << mytime6 - mytime2;
    //cout << endl << "Anh can tim: " << possition << " with score " << myMap[possition] << endl;
    file.close();
    w_file.close();
}

int main() {
    float runtime1, runtime2;
    runtime1 = getTime();
    //readFile();
    go();
    runtime2 = getTime();

    cout << endl << endl << "Tong thoi gian: " << runtime2 - runtime1 << endl;
    system("pause");
