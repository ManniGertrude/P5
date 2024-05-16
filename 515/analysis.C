#define analysis_cxx
#include "analysis.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TRint.h>

void analysis::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L analysis.C
//      Root > analysis t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch

  TH1D* Winkel = new TH1D("1", "Winkelverteilung", 280,-70, 70);
  TH2 *Histo = new TH2D("2", "Abstandssumme gegen Abstandsdifferenz",80,-8.5,8.5, 80, 0, 17);
   TH1D* DT = new TH1D("3", "Treffer pro Driftzeit", 251,0,627.5);
   TH1D* ODB = new TH1D("4", "Orts-Driftzeitbeziehung", 251,0,627.5);

   if (fChain == 0) return;


   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      
      for(UInt_t hit=0; hit<nhits_le; hit++) {
        if (tot[hit] < 16) break;
        if (0.884615* time_le[hit] > tot[hit]+56*0.884615)break;
        if(wire_le[hit] % 2 == 0){wire_le[hit]--;} else{wire_le[hit]++;}
        
        Double_t time=time_le[hit];
        DT->Fill(time);

        Double_t sum=0;
        for(UInt_t bin=1; bin <=DT->GetNbinsX(); ++bin){
          sum += DT->GetBinContent(bin);
          ODB->SetBinContent(bin,sum);
        } 
        ODB->Scale(8.5/sum);

        for (UInt_t j=0; j<=nhits_le; j++) {
          if(j+1<=nhits_le){



          if(wire_le[j] <= 15){
            Winkel->Fill(atan((15-wire_le[j])*8.5/134)*57.2957795);
            ;}
          else if (wire_le[j] > 15){
            Winkel->Fill(-atan((wire_le[j]-15)*8.5/134)*57.2957795);
          ;}
            if(wire_le[j] == wire_le[j+1]-1){
              if(6 <= wire_le[j] && wire_le[j] <=10){
                Histo->Fill(ODB->GetBinContent(time_le[j+1]) - ODB->GetBinContent(time_le[j]),ODB->GetBinContent(time_le[j+1]) + ODB->GetBinContent(time_le[j]) );
                Histo->Fill(ODB->GetBinContent(time_le[j]) - ODB->GetBinContent(time_le[j+1]),ODB->GetBinContent(time_le[j+1]) + ODB->GetBinContent(time_le[j]) );
              ;}
            ;}
          }
        }
      }
    }
   Winkel->GetXaxis()->SetTitle("Winkel in #circ");
   Winkel->GetYaxis()->SetTitle("Trefferanzahl");
   Histo->GetXaxis()->SetTitle("Abstandsdifferenz in mm");
   Histo->GetYaxis()->SetTitle("Abstandssumme in mm");
   gStyle->SetOptStat(0);
   gStyle->SetPalette(107);
   //Winkel->Fit("gaus");
   //Winkel->Draw();
   Histo->Draw("colz");
  

  }


int main(int argc, char** argv) {
  TROOT root("app","app");
  Int_t dargc=1;
  char** dargv = &argv[0];
  TRint *app = new TRint("app", &dargc, dargv);
  //TRint *app = new TRint("app", 0, NULL);
  TCanvas *c1 = new TCanvas("c", "c", 800, 600);
  TFile *f=new TFile(argv[1]);
  TTree *tree=(TTree*)f->FindObjectAny("t");
  //tree->Dump();
  analysis* ana = new analysis(tree);
  ana->Loop();
 
  app->Run(kTRUE);
}