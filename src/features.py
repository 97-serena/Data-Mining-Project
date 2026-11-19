"""Input schema for scoring customers before retention outreach.

Column types describe how values are encoded. For example, the numeric codes
in IncomeGroup represent categories.
"""

NUMERIC_COLUMNS = (
    "MonthlyRevenue",
    "MonthlyMinutes",
    "TotalRecurringCharge",
    "DirectorAssistedCalls",
    "OverageMinutes",
    "RoamingCalls",
    "PercChangeMinutes",
    "PercChangeRevenues",
    "DroppedCalls",
    "BlockedCalls",
    "UnansweredCalls",
    "CustomerCareCalls",
    "ThreewayCalls",
    "ReceivedCalls",
    "OutboundCalls",
    "InboundCalls",
    "PeakCallsInOut",
    "OffPeakCallsInOut",
    "DroppedBlockedCalls",
    "CallForwardingCalls",
    "CallWaitingCalls",
    "MonthsInService",
    "UniqueSubs",
    "ActiveSubs",
    "Handsets",
    "HandsetModels",
    "CurrentEquipmentDays",
    "AgeHH1",
    "AgeHH2",
    "ReferralsMadeBySubscriber",
    "AdjustmentsToCreditRating",
)

BINARY_COLUMNS = (
    "ChildrenInHH",
    "HandsetRefurbished",
    "HandsetWebCapable",
    "TruckOwner",
    "RVOwner",
    "BuysViaMailOrder",
    "RespondsToMailOffers",
    "OptOutMailings",
    "NonUSTravel",
    "OwnsComputer",
    "HasCreditCard",
    "NewCellphoneUser",
    "NotNewCellphoneUser",
    "OwnsMotorcycle",
)

SERVICE_AREA = "ServiceArea"
CATEGORICAL_COLUMNS = (
    "Homeownership",
    "IncomeGroup",
    "HandsetPrice",
    "CreditRating",
    "PrizmCode",
    "Occupation",
    "MaritalStatus",
)

# RetentionCalls, RetentionOffersAccepted and MadeCallToRetentionTeam have no
# verified timing relative to outreach. CustomerID and the outcome are not inputs.
PREDICTOR_COLUMNS = (*NUMERIC_COLUMNS, *BINARY_COLUMNS, SERVICE_AREA, *CATEGORICAL_COLUMNS)
